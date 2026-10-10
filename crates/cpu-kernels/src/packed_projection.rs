//! Constant output-channel packing for portable runtime-dispatched nano-gemm.
//! Packing moves data only; all products execute in the library's kernels.
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

const BLOCK: usize = 32;

// Only created with the runtime-dispatched public Plan constructor. Its mask
// pointers refer to nano-gemm 0.2.2's constant architecture tables; all other
// fields are dimensions/strides and static kernel function pointers. No caller
// buffers or mutable execution state are retained. Pin the dependency before
// relying on this audited constructor representation.
#[derive(Clone, Copy)]
struct LibraryPlan(nano_gemm::Plan<f32>);
// SAFETY: plans made by the constructor above contain immutable/static data
// only; execute_unchecked takes &self and each caller supplies separate output.
unsafe impl Send for LibraryPlan {}
unsafe impl Sync for LibraryPlan {}
impl std::fmt::Debug for LibraryPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LibraryPlan")
    }
}
impl LibraryPlan {
    fn kernel(&self) -> &nano_gemm::Plan<f32> {
        &self.0
    }
}
#[derive(Debug)]
struct RowPlans {
    full: LibraryPlan,
    tail: LibraryPlan,
}

type PlanCache = Arc<Mutex<Vec<(usize, Arc<RowPlans>)>>>;

/// Immutable F32 weights packed in small output-channel blocks.
#[derive(Clone, Debug)]
pub struct PackedProjection {
    input: usize,
    output: usize,
    weights: Arc<Vec<f32>>,
    plans: PlanCache,
}

impl PackedProjection {
    /// Prepare row-major `(output, input)` weights independently of their source.
    pub fn new(weights: &[f32], input: usize, output: usize) -> Result<Self, String> {
        if input == 0 || output == 0 || input.checked_mul(output) != Some(weights.len()) {
            return Err("invalid packed projection weight shape".into());
        }
        let extent = output
            .div_ceil(BLOCK)
            .checked_mul(BLOCK)
            .and_then(|n| n.checked_mul(input))
            .ok_or("packed weight extent overflow")?;
        if input > isize::MAX as usize || output > isize::MAX as usize {
            return Err("packed projection strides exceed isize".into());
        }
        let mut packed = vec![0.; extent];
        packed
            .par_chunks_mut(input * BLOCK)
            .enumerate()
            .for_each(|(block, dst)| {
                let begin = block * BLOCK;
                for i in 0..input {
                    for channel in 0..BLOCK.min(output - begin) {
                        dst[i * BLOCK + channel] = weights[(begin + channel) * input + i];
                    }
                }
            });
        Ok(Self {
            input,
            output,
            weights: Arc::new(packed),
            plans: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Packed weight bytes (including channel padding) and bounded plan metadata.
    pub fn retained_bytes(&self) -> usize {
        let metadata = self
            .plans
            .lock()
            .map(|plans| {
                plans.len()
                    * (std::mem::size_of::<RowPlans>()
                        + std::mem::size_of::<(usize, Arc<RowPlans>)>())
            })
            .unwrap_or(0);
        self.weights.capacity() * std::mem::size_of::<f32>() + metadata
    }

    /// Feature-first column-major `y = weight * x`, or accumulate into `y`.
    /// Input has `(input, rows)` and output `(output, rows)` storage.
    pub fn run_into(
        &self,
        x: &[f32],
        rows: usize,
        y: &mut [f32],
        accumulate: bool,
    ) -> Result<(), String> {
        if self.input.checked_mul(rows) != Some(x.len())
            || self.output.checked_mul(rows) != Some(y.len())
        {
            return Err("invalid packed projection activation/output shape".into());
        }
        if rows == 0 {
            return Ok(());
        }
        if x.len() > isize::MAX as usize || y.len() > isize::MAX as usize {
            return Err("packed projection extent exceeds isize".into());
        }
        let destination = y.as_mut_ptr() as usize;
        let blocks = self.output.div_ceil(BLOCK);
        let plans = {
            let mut cache = self
                .plans
                .lock()
                .map_err(|_| "packed plan cache lock poisoned")?;
            if let Some((_, plans)) = cache.iter().find(|(length, _)| *length == rows) {
                plans.clone()
            } else {
                let tail = if self.output % BLOCK == 0 {
                    BLOCK
                } else {
                    self.output % BLOCK
                };
                let plans = Arc::new(RowPlans {
                    full: LibraryPlan(nano_gemm::Plan::new_colmajor_lhs_and_dst_f32(
                        BLOCK, rows, self.input,
                    )),
                    tail: LibraryPlan(nano_gemm::Plan::new_colmajor_lhs_and_dst_f32(
                        tail, rows, self.input,
                    )),
                });
                if cache.len() == 32 {
                    cache.remove(0);
                }
                cache.push((rows, plans.clone()));
                plans
            }
        };
        (0..blocks).into_par_iter().for_each(|block| {
            let begin = block * BLOCK;
            let channels = BLOCK.min(self.output - begin);
            let plan = if channels == BLOCK {
                plans.full.kernel()
            } else {
                plans.tail.kernel()
            };
            // SAFETY: each task owns a disjoint output-channel band in every
            // token. The packed source includes a full padded BLOCK for tails;
            // the plan's masked tail accesses/writes only `channels` outputs.
            // All strides describe the validated buffers. Plans and sources
            // are immutable for execution.
            unsafe {
                plan.execute_unchecked(
                    channels,
                    rows,
                    self.input,
                    (destination as *mut f32).add(begin),
                    1,
                    self.output as isize,
                    self.weights.as_ptr().add(block * self.input * BLOCK),
                    1,
                    BLOCK as isize,
                    x.as_ptr(),
                    1,
                    self.input as isize,
                    if accumulate { 1. } else { 0. },
                    1.,
                    false,
                    false,
                );
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_projection_handles_tails_accumulation_and_plan_eviction() {
        for workers in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            pool.install(|| {
                for output in [1, 3, 15, 16, 17, 31, 32, 33, 64, 70] {
                    let input = 35;
                    let weights: Vec<f32> = (0..input * output)
                        .map(|i| (i % 19) as f32 * 0.02 - 0.18)
                        .collect();
                    let packed = PackedProjection::new(&weights, input, output).unwrap();
                    for rows in (0..35).chain([1, 8, 65]) {
                        let x: Vec<f32> = (0..input * rows)
                            .map(|i| (i % 23) as f32 * 0.01 - 0.11)
                            .collect();
                        for accumulate in [false, true] {
                            let mut expected = vec![0.25; output * rows];
                            for token in 0..rows {
                                for channel in 0..output {
                                    let mut sum = if accumulate { 0.25 } else { 0. };
                                    for i in 0..input {
                                        sum += weights[channel * input + i] * x[token * input + i];
                                    }
                                    expected[token * output + channel] = sum;
                                }
                            }
                            let mut actual =
                                vec![if accumulate { 0.25 } else { f32::NAN }; output * rows];
                            packed.run_into(&x, rows, &mut actual, accumulate).unwrap();
                            for (a, b) in actual.iter().zip(expected) {
                                assert!((a - b).abs() < 2e-5, "{a} vs {b}");
                            }
                        }
                    }
                    assert!(packed.plans.lock().unwrap().len() <= 32);
                }
            });
        }
    }

    #[test]
    fn owned_packing_is_shared_safely_across_concurrent_calls() {
        let weights: Vec<f32> = (0..7 * 70).map(|i| (i % 13) as f32 * 0.01).collect();
        let projection = Arc::new(PackedProjection::new(&weights, 7, 70).unwrap());
        drop(weights);
        let x = vec![0.5; 7 * 3];
        let mut expected = vec![0.; 70 * 3];
        projection.run_into(&x, 3, &mut expected, false).unwrap();
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let projection = projection.clone();
                let x = x.clone();
                std::thread::spawn(move || {
                    let mut actual = vec![0.; 70 * 3];
                    projection.run_into(&x, 3, &mut actual, false).unwrap();
                    actual
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
        assert!(
            projection
                .run_into(&[0.; 6], 1, &mut [0.; 70], false)
                .is_err()
        );
        assert!(PackedProjection::new(&[], 0, 1).is_err());
        assert!(PackedProjection::new(&[0.; 1], 1, 2).is_err());
    }
}

//! Owned reusable strict-F32 projection using system oneDNN primitives.
//!
//! This module supplies preparation, not a process-wide pointer cache. The
//! owner controls its lifetime and can place it behind a mutex when sharing.
use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn decision_projection_create(
        weights: *const f32,
        input: i32,
        output: i32,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn decision_projection_run(
        handle: *mut c_void,
        x: *const f32,
        rows: i32,
        y: *mut f32,
        accumulate: bool,
        scratch: *mut c_void,
        scratch_capacity: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn decision_projection_scratch_bytes(
        handle: *mut c_void,
        rows: i32,
        accumulate: bool,
        bytes: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn decision_projection_geglu_bytes(
        handle: *mut c_void,
        rows: i32,
        bytes: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn decision_projection_geglu(
        handle: *mut c_void,
        projected: *const f32,
        rows: i32,
        result: *mut f32,
        workspace: *mut c_void,
        workspace_capacity: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn decision_norm_create(
        rows: i32,
        width: i32,
        epsilon: f32,
        bias: bool,
        scratch_bytes: *mut usize,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn decision_norm_run(
        handle: *mut c_void,
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        y: *mut f32,
        workspace: *mut c_void,
        workspace_capacity: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> i32;
    fn decision_norm_destroy(handle: *mut c_void);
    fn decision_projection_destroy(handle: *mut c_void);
    fn decision_projection_bytes(handle: *const c_void) -> usize;
}

/// An owned CPU projection. Packed weights are independent of the source slice.
/// Execution requires exclusive access to its plans, stream and scratch state.
pub struct PreparedProjection {
    handle: NonNull<c_void>,
    input: usize,
    output: usize,
    scratch: ProjectionWorkspace,
}

// SAFETY: CPU engine/primitive/memory handles may be used on another thread.
// Every retained primitive uses user-owned scratchpad memory, avoiding the
// thread-affine library scratchpad policy. All execution requires &mut self;
// callers sharing an instance must provide synchronization. It is not Sync.
unsafe impl Send for PreparedProjection {}

impl std::fmt::Debug for PreparedProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedProjection")
            .field("input", &self.input)
            .field("output", &self.output)
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}

fn error_text(error: &[c_char; 512]) -> String {
    // SAFETY: the C++ boundary always terminates its error message and the
    // buffer starts fully zeroed, including for an empty message.
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl PreparedProjection {
    /// Pack row-major `(output, input)` weights. Shapes must be nonempty and
    /// fit oneDNN's C wrapper integers; products are checked before FFI calls.
    pub fn new(weights: &[f32], input: usize, output: usize) -> Result<Self, String> {
        let extent = input.checked_mul(output).ok_or("weight extent overflow")?;
        if input == 0 || output == 0 || weights.len() != extent {
            return Err("invalid projection weight shape".into());
        }
        let input_c = i32::try_from(input).map_err(|_| "input dimension exceeds i32")?;
        let output_c = i32::try_from(output).map_err(|_| "output dimension exceeds i32")?;
        let mut error = [0; 512];
        // SAFETY: the checked slice covers all weights. The wrapper copies
        // them before returning and catches all C++ exceptions.
        let handle = unsafe {
            decision_projection_create(
                weights.as_ptr(),
                input_c,
                output_c,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| error_text(&error))?;
        Ok(Self {
            handle,
            input,
            output,
            scratch: ProjectionWorkspace::default(),
        })
    }

    /// Packed weight and execution scratchpad bytes owned by this instance.
    /// Includes distinct formats for sequence lengths; excludes library JIT code.
    pub fn retained_bytes(&self) -> usize {
        // SAFETY: handle remains valid until Drop and the getter is read-only.
        unsafe { decision_projection_bytes(self.handle.as_ptr()) }
            .saturating_add(self.scratch.retained_bytes())
    }

    /// Compute `y = x · weightᵀ`, or accumulate that product into `y`.
    /// Activations and outputs are row-major `(rows, features)`.
    pub fn run(
        &mut self,
        x: &[f32],
        rows: usize,
        y: &mut [f32],
        accumulate: bool,
    ) -> Result<(), String> {
        let mut workspace = std::mem::take(&mut self.scratch);
        let result = self.run_with_workspace(x, rows, y, accumulate, &mut workspace);
        self.scratch = workspace;
        result
    }

    /// Execute using an exclusive workspace reusable across different prepared
    /// projections. The workspace owns no runtime or projection handles.
    pub fn run_with_workspace(
        &mut self,
        x: &[f32],
        rows: usize,
        y: &mut [f32],
        accumulate: bool,
        workspace: &mut ProjectionWorkspace,
    ) -> Result<(), String> {
        if Some(x.len()) != rows.checked_mul(self.input)
            || Some(y.len()) != rows.checked_mul(self.output)
        {
            return Err("invalid projection activation/output shape".into());
        }
        if rows == 0 {
            return Ok(());
        }
        let rows = i32::try_from(rows).map_err(|_| "row count exceeds i32")?;
        let mut error = [0; 512];
        let mut required = 0;
        // SAFETY: the handle is exclusively borrowed and the scalar output is valid.
        let status = unsafe {
            decision_projection_scratch_bytes(
                self.handle.as_ptr(),
                rows,
                accumulate,
                &mut required,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(error_text(&error));
        }
        workspace.reserve(required)?;
        // SAFETY: checked extents cover both buffers; exclusive self and y
        // prevent races. Execution waits before releasing the borrowed data.
        let status = unsafe {
            decision_projection_run(
                self.handle.as_ptr(),
                x.as_ptr(),
                rows,
                y.as_mut_ptr(),
                accumulate,
                workspace.pointer(),
                workspace.retained_bytes(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(error_text(&error))
        }
    }
    /// Apply erf-based GELU and gate multiplication through strict-F32 oneDNN
    /// primitives. Projected data contains interleaved value/gate halves per row.
    pub fn geglu_with_workspace(
        &mut self,
        projected: &[f32],
        rows: usize,
        result: &mut [f32],
        workspace: &mut ProjectionWorkspace,
    ) -> Result<(), String> {
        if self.output % 2 != 0
            || rows.checked_mul(self.output) != Some(projected.len())
            || rows.checked_mul(self.output / 2) != Some(result.len())
        {
            return Err("invalid GeGLU shape".into());
        }
        if rows == 0 {
            return Ok(());
        }
        let rows = i32::try_from(rows).map_err(|_| "row count exceeds i32")?;
        let mut error = [0; 512];
        let mut required = 0;
        // SAFETY: the handle is exclusive and scalar outputs are valid.
        let status = unsafe {
            decision_projection_geglu_bytes(
                self.handle.as_ptr(),
                rows,
                &mut required,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(error_text(&error));
        }
        workspace.reserve(required)?;
        // SAFETY: validated extents, exclusive output/workspace and synchronous
        // primitives guarantee all borrowed buffers remain valid until return.
        let status = unsafe {
            decision_projection_geglu(
                self.handle.as_ptr(),
                projected.as_ptr(),
                rows,
                result.as_mut_ptr(),
                workspace.pointer(),
                workspace.retained_bytes(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(error_text(&error))
        }
    }
}

impl Drop for PreparedProjection {
    fn drop(&mut self) {
        // SAFETY: this instance exclusively owns one live C++ allocation.
        unsafe { decision_projection_destroy(self.handle.as_ptr()) }
    }
}

/// Aligned high-water scratch storage shared by sequential CPU projections.
#[derive(Debug, Default)]
pub struct ProjectionWorkspace {
    allocation: Option<(NonNull<u8>, std::alloc::Layout)>,
}
// SAFETY: the allocation is exclusively owned; mutation requires &mut self,
// and shared methods only inspect metadata, never the mutable scratch bytes.
unsafe impl Send for ProjectionWorkspace {}
// SAFETY: no shared reference exposes or modifies scratch storage.
unsafe impl Sync for ProjectionWorkspace {}
impl ProjectionWorkspace {
    /// Allocated scratch bytes, including alignment/capacity padding.
    pub fn retained_bytes(&self) -> usize {
        self.allocation
            .as_ref()
            .map_or(0, |(_, layout)| layout.size())
    }
    fn pointer(&mut self) -> *mut c_void {
        self.allocation
            .as_mut()
            .map_or(std::ptr::null_mut(), |(p, _)| p.as_ptr().cast())
    }
    fn reserve(&mut self, required: usize) -> Result<(), String> {
        if required <= self.retained_bytes() {
            return Ok(());
        }
        let capacity = required
            .checked_next_power_of_two()
            .ok_or("scratchpad extent overflow")?;
        let layout = std::alloc::Layout::from_size_align(capacity, 64)
            .map_err(|_| "invalid scratchpad layout")?;
        // SAFETY: valid nonzero layout. Ownership is paired with exactly this layout.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .ok_or("scratchpad allocation failed")?;
        if let Some((old, old_layout)) = self.allocation.replace((pointer, layout)) {
            // SAFETY: old pointer was allocated with old_layout and is no longer used.
            unsafe { std::alloc::dealloc(old.as_ptr(), old_layout) };
        }
        Ok(())
    }
}
impl Drop for ProjectionWorkspace {
    fn drop(&mut self) {
        if let Some((pointer, layout)) = self.allocation.take() {
            // SAFETY: allocation is uniquely owned and native execution waits before returning.
            unsafe { std::alloc::dealloc(pointer.as_ptr(), layout) };
        }
    }
}

/// Strict-F32 LayerNorm plan. Inputs remain borrowed only during execution.
#[derive(Debug)]
pub struct PreparedLayerNorm {
    handle: NonNull<c_void>,
    rows: usize,
    width: usize,
    bias: bool,
    scratch_bytes: usize,
}
// SAFETY: user-mode scratchpads remove thread affinity; every execution is
// exclusive and synchronous, and shared references only inspect metadata.
unsafe impl Send for PreparedLayerNorm {}
// SAFETY: mutation of the native handles requires &mut self.
unsafe impl Sync for PreparedLayerNorm {}
impl PreparedLayerNorm {
    /// Prepare normalization of row-major `(rows,width)` activations.
    pub fn new(rows: usize, width: usize, epsilon: f32, bias: bool) -> Result<Self, String> {
        if rows == 0 || width == 0 || !epsilon.is_finite() || epsilon <= 0. {
            return Err("invalid LayerNorm configuration".into());
        }
        rows.checked_mul(width)
            .ok_or("normalization extent overflow")?;
        let rows_c = i32::try_from(rows).map_err(|_| "row count exceeds i32")?;
        let width_c = i32::try_from(width).map_err(|_| "width exceeds i32")?;
        let mut error = [0; 512];
        let mut scratch_bytes = 0;
        // SAFETY: validated scalar dimensions, exclusive valid output pointers;
        // constructor catches C++ exceptions and owns all native resources.
        let handle = unsafe {
            decision_norm_create(
                rows_c,
                width_c,
                epsilon,
                bias,
                &mut scratch_bytes,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| error_text(&error))?;
        Ok(Self {
            handle,
            rows,
            width,
            bias,
            scratch_bytes,
        })
    }
    /// Execute with affine parameters and an exclusive reusable workspace.
    pub fn run(
        &mut self,
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        y: &mut [f32],
        workspace: &mut ProjectionWorkspace,
    ) -> Result<(), String> {
        let extent = self.rows * self.width;
        if x.len() != extent
            || y.len() != extent
            || weight.len() != self.width
            || bias.is_some() != self.bias
            || bias.is_some_and(|b| b.len() != self.width)
        {
            return Err("invalid LayerNorm input shape".into());
        }
        workspace.reserve(self.scratch_bytes)?;
        let mut error = [0; 512];
        // SAFETY: validated extents, exclusive mutable handles/buffers. Native
        // execution waits and clears every data handle before returning.
        let status = unsafe {
            decision_norm_run(
                self.handle.as_ptr(),
                x.as_ptr(),
                weight.as_ptr(),
                bias.map_or(std::ptr::null(), |b| b.as_ptr()),
                y.as_mut_ptr(),
                workspace.pointer(),
                workspace.retained_bytes(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(error_text(&error))
        }
    }
}
impl Drop for PreparedLayerNorm {
    fn drop(&mut self) {
        // SAFETY: one owned live native allocation, with no pending execution.
        unsafe { decision_norm_destroy(self.handle.as_ptr()) };
    }
}

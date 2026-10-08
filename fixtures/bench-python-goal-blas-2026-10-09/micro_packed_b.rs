//! Diagnostic only: compare current GEMM with a separately supplied MKL library.
use std::ffi::{CString, c_char, c_void};
use std::time::Instant;
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
type Sgemm = unsafe extern "C" fn(*const c_char,*const c_char,*const i32,*const i32,*const i32,*const f32,*const f32,*const i32,*const f32,*const i32,*const f32,*mut f32,*const i32);
type PackSize = unsafe extern "C" fn(i32,i32,i32,i32) -> usize;
type Pack = unsafe extern "C" fn(i32,i32,i32,i32,i32,i32,f32,*const f32,i32,*mut f32);
type Compute = unsafe extern "C" fn(i32,i32,i32,i32,i32,i32,*const f32,i32,*const f32,i32,f32,*mut f32,i32);
type Threads = unsafe extern "C" fn(i32) -> i32;
fn median(values: &mut [f64]) -> f64 { values.sort_by(f64::total_cmp); values[values.len()/2] }
fn main() {
    let path = CString::new(std::env::args().nth(1).expect("MKL shared library")).unwrap();
    let handle = unsafe { dlopen(path.as_ptr(), 2) };
    assert!(!handle.is_null(), "could not load library");
    let lookup = |name: &str| { let name = CString::new(name).unwrap(); let ptr = unsafe { dlsym(handle, name.as_ptr()) }; assert!(!ptr.is_null(), "missing symbol"); ptr };
    let gemm: Sgemm = unsafe { std::mem::transmute(lookup("sgemm_")) };
    let threads: Threads = unsafe { std::mem::transmute(lookup("MKL_Set_Num_Threads_Local")) };
    unsafe { threads(8); }
    let size: PackSize = unsafe { std::mem::transmute(lookup("cblas_sgemm_pack_get_size")) };
    let pack: Pack = unsafe { std::mem::transmute(lookup("cblas_sgemm_pack")) };
    let compute: Compute = unsafe { std::mem::transmute(lookup("cblas_sgemm_compute")) };
    for rows in [8usize,16,64] {
        for (input,output) in [(1024usize,1024usize),(1024,3072),(1024,5248),(2624,1024)] {
            let x: Vec<f32> = (0..rows*input).map(|i| (i%97) as f32 * 0.001-0.048).collect();
            let w: Vec<f32> = (0..input*output).map(|i| (i%89) as f32 * 0.002-0.088).collect();
            let mut old=vec![0.0f32;rows*output]; let mut new=old.clone(); let mut packed_output=old.clone();
            let layout = std::alloc::Layout::from_size_align(unsafe { size(162,rows as i32,output as i32,input as i32) },64).unwrap();
            let packed = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
            assert!(!packed.is_null());
            let pack_start=Instant::now();
            unsafe { pack(101,162,112,rows as i32,output as i32,input as i32,1.0,w.as_ptr(),input as i32,packed); }
            let pack_ms=pack_start.elapsed().as_secs_f64()*1000.;
            let run_packed = |y: &mut [f32]| unsafe { compute(101,111,151,rows as i32,output as i32,input as i32,x.as_ptr(),input as i32,packed,0,0.0,y.as_mut_ptr(),output as i32); };
            let run_mkl = |y: &mut [f32]| unsafe {
                gemm(c"T".as_ptr(),c"N".as_ptr(),&(output as i32),&(rows as i32),&(input as i32),&1.0,w.as_ptr(),&(input as i32),x.as_ptr(),&(input as i32),&0.0,y.as_mut_ptr(),&(output as i32));
            };
            let mut old_ms=Vec::new(); let mut new_ms=Vec::new(); let mut packed_ms=Vec::new();
            for iteration in 0..20 {
                for candidate in if iteration%2==0 {[0,1,2]} else {[2,1,0]} {
                    let start=Instant::now();
                    match candidate {1=>run_mkl(&mut new),2=>run_packed(&mut packed_output),_=>cpu_kernels::input_mul_weight_transpose_into(&x,rows,input,&w,output,&mut old)}
                    let ms=start.elapsed().as_secs_f64()*1000.;
                    if iteration>=5 {match candidate {1=>new_ms.push(ms),2=>packed_ms.push(ms),_=>old_ms.push(ms)}}
                }
            }
            let error=old.iter().zip(&new).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            assert!(error < 1e-4, "error={error}");
            let packed_error=old.iter().zip(&packed_output).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            assert!(packed_error<1e-4, "packed_error={packed_error}");
            println!("rows={rows} in={input} out={output} baseline_ms={:.4} mkl_ms={:.4} max_error={error}",median(&mut old_ms),median(&mut new_ms));
            println!("packed_ms={:.4} pack_ms={pack_ms:.4} packed_bytes={} packed_error={packed_error}",median(&mut packed_ms),layout.size());
            unsafe { std::alloc::dealloc(packed as *mut u8,layout); }
        }
    }
}

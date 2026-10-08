//! Diagnostic only: compare current GEMM with a separately supplied MKL library.
use std::ffi::{CString, c_char, c_void};
use std::time::Instant;
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
}
type PackSize = unsafe extern "C" fn(i32,i32,i32,i32) -> usize;
type Pack = unsafe extern "C" fn(i32,i32,i32,i32,i32,i32,f32,*const f32,i32,*mut f32);
type Compute = unsafe extern "C" fn(i32,i32,i32,i32,i32,i32,*const f32,i32,*const f32,i32,f32,*mut f32,i32);
type Threads = unsafe extern "C" fn(i32) -> i32;
fn median(values: &mut [f64]) -> f64 { values.sort_by(f64::total_cmp); values[values.len()/2] }

use rayon::prelude::*;
struct Packed { address: usize, layout: std::alloc::Layout, offset: usize, width: usize }
impl Drop for Packed { fn drop(&mut self) { unsafe { std::alloc::dealloc(self.address as *mut u8,self.layout); } } }
fn main() {
 let path=CString::new(std::env::args().nth(1).expect("library")).unwrap();
 let handle=unsafe{dlopen(path.as_ptr(),2)}; assert!(!handle.is_null());
 let lookup=|name:&str|{let n=CString::new(name).unwrap();let p=unsafe{dlsym(handle,n.as_ptr())};assert!(!p.is_null());p};
 let threads:Threads=unsafe{std::mem::transmute(lookup("MKL_Set_Num_Threads_Local"))};
 let size:PackSize=unsafe{std::mem::transmute(lookup("cblas_sgemm_pack_get_size"))};
 let pack:Pack=unsafe{std::mem::transmute(lookup("cblas_sgemm_pack"))};
 let compute:Compute=unsafe{std::mem::transmute(lookup("cblas_sgemm_compute"))};
 for (input,output) in [(1024usize,1024usize),(1024,3072),(1024,5248),(2624,1024)] {
  let w:Vec<f32>=(0..input*output).map(|i|(i%89) as f32*0.002-0.088).collect();
  for target in [256usize*1024,768*1024] {
   let block=(target/(input*4)/16*16).max(16);
   let old_threads=unsafe{threads(1)};
   let start=Instant::now();
   let packed:Vec<Packed>=(0..output).step_by(block).map(|offset|{
    let width=block.min(output-offset);
    let layout=std::alloc::Layout::from_size_align(unsafe{size(161,width as i32,32,input as i32)},64).unwrap();
    let address=unsafe{std::alloc::alloc_zeroed(layout)} as usize;assert_ne!(address,0);
    unsafe{pack(102,161,112,width as i32,32,input as i32,1.0,w.as_ptr().add(offset*input),input as i32,address as *mut f32)};
    Packed{address,layout,offset,width}
   }).collect();
   unsafe{threads(old_threads)};
   let pack_ms=start.elapsed().as_secs_f64()*1000.;
   for rows in [1usize,3,7,8,9,16,64,65] {
    let x:Vec<f32>=(0..rows*input).map(|i|(i%97)as f32*0.001-0.048).collect();
    let mut padded=x.clone();padded.resize(rows.div_ceil(32)*32*input,0.0);
    let mut baseline=vec![0.;rows*output];let mut result=baseline.clone();
    let run=|out:&mut[f32]|{
     // Each task owns distinct output features for every token. Tail tokens
     // use a private buffer, so the fixed packed tile=32 never overruns output.
     let address=out.as_mut_ptr() as usize;
     packed.par_iter().for_each(|p|{
      let prior=unsafe{threads(1)};
      for row in (0..rows).step_by(32){
       if rows-row>=32 {
        unsafe{compute(102,151,111,p.width as i32,32,input as i32,p.address as *const f32,0,padded.as_ptr().add(row*input),input as i32,0.,(address as *mut f32).add(row*output+p.offset),output as i32)};
       } else {
        let mut tail=vec![0.;32*p.width];
        unsafe{compute(102,151,111,p.width as i32,32,input as i32,p.address as *const f32,0,padded.as_ptr().add(row*input),input as i32,0.,tail.as_mut_ptr(),p.width as i32)};
        for t in 0..rows-row {unsafe{std::ptr::copy_nonoverlapping(tail.as_ptr().add(t*p.width),(address as *mut f32).add((row+t)*output+p.offset),p.width)}}
       }
      }
      unsafe{threads(prior)};
     });
    };
    let mut base_ms=Vec::new();let mut packed_ms=Vec::new();
    for iteration in 0..20 {
     for candidate in if iteration%2==0 {[0,1]}else{[1,0]} {
      let start=Instant::now();
      if candidate==0 {cpu_kernels::input_mul_weight_transpose_into(&x,rows,input,&w,output,&mut baseline)}else{run(&mut result)}
      let ms=start.elapsed().as_secs_f64()*1000.;
      if iteration>=5 {if candidate==0 {base_ms.push(ms)}else{packed_ms.push(ms)}}
     }
    }
    let error=baseline.iter().zip(&result).map(|(a,b)|(a-b).abs()).fold(0.,f32::max);assert!(error<1e-4,"rows={rows} error={error}");
    println!("in={input} out={output} block={block} rows={rows} baseline_ms={:.4} packed_ms={:.4} max_error={error} pack_ms={pack_ms:.3} packed_bytes={}",median(&mut base_ms),median(&mut packed_ms),packed.iter().map(|p|p.layout.size()).sum::<usize>());
   }
  }
 }
}

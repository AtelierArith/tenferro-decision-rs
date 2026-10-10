import json, os, pathlib, subprocess, sys, time
output=pathlib.Path(sys.argv[1]); command=sys.argv[2:]
hz=os.sysconf('SC_CLK_TCK')
def processes():
    result={}
    for directory in pathlib.Path('/proc').iterdir():
        if not directory.name.isdigit(): continue
        try:
            value=(directory/'stat').read_text(); fields=value[value.rfind(')')+2:].split()
            result[int(directory.name)]=(int(fields[11])+int(fields[12]), int(fields[1]), value[value.find('(')+1:value.rfind(')')])
        except (OSError, ValueError): pass
    return result
with output.open('w') as target:
    child=subprocess.Popen(command, stdout=target)
    before=processes(); stamp=time.monotonic(); started=stamp
    observations=[]; max_other=0.0; interval_count=0
    while child.poll() is None:
        time.sleep(0.5)
        after=processes(); now=time.monotonic(); duration=now-stamp
        owned={child.pid,os.getpid()}
        while True:
            extra={pid for pid,(_,parent,_) in after.items() if parent in owned}
            if extra.issubset(owned): break
            owned.update(extra)
        busy=[]; other=0.0
        for pid,(ticks,_,name) in after.items():
            if pid in owned or pid not in before: continue
            cpu=max(0,ticks-before[pid][0])/hz/duration
            other+=cpu
            if cpu>=0.25: busy.append({'pid':pid,'process':name,'cpu_cores':round(cpu,3)})
        max_other=max(max_other,other); interval_count+=1
        if busy: observations.append({'elapsed_s':round(now-started,3),'processes':busy})
        before=after; stamp=now
    code=child.wait()
report={'exit_code':code,'duration_s':round(time.monotonic()-started,3),'poll_interval_s':0.5,'observations':interval_count,'heavy_other_process_threshold_cores':0.25,'max_other_cpu_cores':round(max_other,3),'heavy_other_process_observations':observations}
output.with_suffix('.environment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'result':output.name,'exit_code':code,'heavy_other_observations':len(observations),'max_other_cpu_cores':round(max_other,3)}))
sys.exit(code)

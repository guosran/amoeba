// RUN: python3 -c 'import importlib.util; p=r"%S/../../../../tools/run-joint-scheduling.py"; s=importlib.util.spec_from_file_location("joint_driver",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); t={"candidate_id":"candidate|0","architecture_sha256":"0000000000000000000000000000000000000000000000000000000000000000","communication_mode":"none","task_schedule":[{"task":"a|b,\n雪","start_cycle":0,"end_cycle":1,"cgra_positions":[{"row":0,"col":0}]}],"dependencies":[],"routes":[]}; print(m._schedule_trace_sha256(t))' | FileCheck %s

// The task name contains delimiter characters, a newline, and multibyte UTF-8.
// Its byte-length-prefixed tuple has one stable cross-language identity.
// CHECK: f0020def805fc787a565bc3c76c0e76fa6988770ee6f630c0844cc4aef3f8ca1

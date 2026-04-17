from vllm.scheduler.sequence import Sequence
from vllm.scheduler.tokenizer import tokenizer
from vllm.scheduler.utils import Config
from vllm.scheduler.utils import SamplingParams
from vllm.scheduler.scheduler import Scheduler
from vllm.scheduler.fake_llm import run_fake_model



config = Config()
config.num_kvcache_blocks = 100
config.max_num_seqs = 3 # 最多运行多少个请求
config.max_model_len = 15 # 单条请求长度


sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
scheduler = Scheduler(config)


# 增加第一个请求
token_ids = tokenizer.encode("Do you subscribe InfraTech?")
seq = Sequence(token_ids, sampling_params)
scheduler.add(seq)

# 增加第二个请求
token_ids = tokenizer.encode("hi, I'm kaiyuan")
seq = Sequence(token_ids, sampling_params)
scheduler.add(seq)

# 打印输入请求情况
print("scheduler waiting queue: ")
for id, seq in enumerate(scheduler.waiting):
    print(f"id:{id} seq:{seq.token_ids}")

# 测试调度与生成：
print("\nrunning: ")
while not scheduler.is_finished():
    seqs, is_prefill = scheduler.schedule()
    token_ids = run_fake_model(seqs, config.max_model_len)
    scheduler.postprocess(seqs, token_ids)
    for id, seq in enumerate(seqs):
        print(f"id:{id} seq:{seq.token_ids}")


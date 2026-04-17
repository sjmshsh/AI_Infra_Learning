from collections import deque
from scheduler.sequence import Sequence
from scheduler.sequence import SequenceStatus       
from scheduler.kv_cache_manager import BlockManager
from scheduler.utils import Config

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 单次批次中允许同时运行的最大序列数量，用于控制并发度
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单次迭代（Iteration）中允许处理的最大 Token 总数，用于控制计算负载。
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_seqs = 0 # 当前batch中实际运行的序列数量
        self.num_batched_tokens = 0 # 本轮GPU需要处理的总token工作量

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def prefill(self):
        scheduled_seqs = []
        while self.waiting and self.num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if self.num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            self.num_seqs += 1
            self.block_manager.allocate(seq)
            # 更新token预算
            self.num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs, True
    
    def decode(self):
        scheduled_seqs = [] # 存储本轮要调度的序列
        # 当有运行中的序列并且没有达到最大序列限制时循环
        while self.running and self.num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 从运行队列头部取出一个序列
            # 检查是否可以给这个序列追加KV缓存块
            while not self.block_manager.can_append(seq):
                # 如果不能追加，需要抢占其他序列
                if self.running:
                    # 抢占运行队列尾部的序列（后进先出）
                    self.preempt(self.running.pop())
                else:
                    # 如果没有其他运行序列，抢占当前序列本身
                    self.preempt(seq)
                    break
            else: # 如果can_append检查通过
                self.num_seqs += 1 # 增加当前批次的序列数量
                self.block_manager.may_append(seq) # 为序列追加KV缓存块
                scheduled_seqs.append(seq) # 添加到本轮调度序列列表
        # 如果有调度成功的序列，将他们重新放回运行队列头部
        if scheduled_seqs:
          self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
    
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def schedule(self, prefill_first=True) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        self.num_seqs = 0
        self.num_batched_tokens = 0
        is_prefill = True

        if prefill_first:
            first_call, second_call = self.prefill, self.decode
        else:
            first_call, second_call = self.decode, self.prefill

        scheduled_seqs, is_prefill = first_call()
        if scheduled_seqs:
            return scheduled_seqs, is_prefill

        scheduled_seqs, is_prefill = second_call()
        if scheduled_seqs:
          return scheduled_seqs, is_prefill
        assert scheduled_seqs

    # 后处理器
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            # 将生成的token添加到对应序列中
            seq.append_token(token_id)
            # 检查序列是否应该结束（遇到EOS token或者达到最大长度）
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # 对于完成的序列，释放KV缓存并从运行队列中移除
                self.block_manager.deallocate(seq)
                self.running.remove(seq)



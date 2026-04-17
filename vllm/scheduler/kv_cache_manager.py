from collections import deque
import xxhash
import numpy as np 
from scheduler.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id # 物理块的唯一标识
        self.ref_count = 0 # 引用计数
        self.hash = -1 # token序列的哈希指纹
        self.token_ids = [] # 存储在块内的实际token id列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size # 每个块能容纳的token数量, 这决定了kv cache的粒度
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 所有物理块的注册表
        self.hash_to_block_id: dict[int, int] = dict() # 从哈希值到物理块ID的映射表
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲块ID的队列
        self.used_block_ids: set[int] = set() # 当前正在被使用的块集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence):  
        assert not seq.block_table
        h = -1
        cache_miss = False
        # 按顺序遍历该序列逻辑上需要的每一个块（例如一个 1000 token 的 Prompt 可能需要 4 个块）
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # 取出第i个逻辑块的token ID列表
            # 这里传入了上一次的哈希值，形成了链式hash
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 拿着算出来的哈希值去hash_to_block_id字典里面找
            block_id = self.hash_to_block_id.get(h, -1)
            # 字典里面没有这个hash，或者产生了哈希碰撞
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 从空闲队列中取出一个物理块ID
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中
                seq.num_cached_tokens += self.block_size
                # 被其他运行中的序列占用，增加引用计数
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 不在used_block_ids当中，在缓存池中但没有使用，说明是冷数据
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)    


    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()


    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

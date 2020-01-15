import argparse
import time
from multiprocessing import Process, Queue, Lock

import matplotlib.pyplot as plt
import numpy as np
import torch

from StreamBuffer import StreamBuffer
from model import VoiceActivityDetector
from utils import load_labeled_audio


def producer(stream_signals, block_size_f, sleep_time, queues):
    n_streams = len(stream_signals)
    ptrs = np.zeros(n_streams, dtype=int)
    prev_time = np.ones(n_streams) * time.perf_counter()

    while True:
        candidates = [i for i in range(n_streams) if ptrs[i] < len(stream_signals[i])]
        if len(candidates) == 0:
            break
        id = candidates[np.random.randint(0, len(candidates))]

        l = ptrs[id]
        r = min(len(stream_signals[id]), l + block_size_f)
        ptrs[id] = r

        block = stream_signals[id][l: r]

        t0 = max(0, sleep_time - (time.perf_counter() - prev_time[id]))
        time.sleep(t0)

        sending_time = time.perf_counter()
        prev_time[id] = sending_time
        queues[id].put((block, sending_time))


def worker(lock, worker_id, queues, buffers, detector, buffer_locks, max_requests, delay_stories, packet_size):
    max_delay = 0
    min_delay = np.inf
    sum_delay = 0

    cnt_requests = 0
    _start = time.perf_counter()

    img_index = 0

    while cnt_requests < max_requests:
        for stream_id, queue in enumerate(queues):
            if queue.empty():
                continue

            block, sending_time = queue.get()

            # START
            with buffer_locks[stream_id]:
                detector.append(block, buffers[stream_id])
                pred = detector.query(buffers[stream_id])
            # END

            finish_time = time.perf_counter()
            delay = finish_time - sending_time

            sum_delay += delay
            cnt_requests += 1

            if delay > max_delay:
                max_delay = delay

            if delay < min_delay:
                min_delay = delay

            delay_stories[stream_id].append (delay)

            with lock:
                print(f'{delay}')

            if cnt_requests % 1000 == 0:
                with lock:
                    print(f'saving graph...')
                    plt.figure(figsize=(15, 10))
                    sz = min(len(story) for story in delay_stories)
                    mn_delay = np.ones(sz, dtype=np.float64) * 1e3
                    mx_delay = np.ones(sz, dtype=np.float64) * (-1e3)
                    mean = np.zeros(sz)
                    bs = []
                    for story in delay_stories:
                        b = np.array(story, dtype=np.float64)[: sz]
                        bs.append(b)
                    bs = np.array(bs)
                    mn_delay = np.min(bs, axis=0)
                    mx_delay = np.max(bs, axis=0)
                    mean_delay = np.sum(bs, axis=0) / bs.shape[0]
                    mean /= len(delay_stories)

                    plt.title(f'{len(delay_stories)} streams, packet size = {packet_size} ms')

                    plt.plot(mn_delay, label='min delay')
                    plt.plot(mx_delay, label='max delay')
                    plt.plot(mean_delay, label='mean delay')
                    plt.ylabel('delay, ms')
                    plt.xlabel('requests number')
                    plt.legend()
                    plt.savefig(f'report/w{worker_id}_{len(delay_stories)}_{img_index}.png')
                    img_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Parallel real time voice activity detector"""
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model is stored'
    )
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to the raw audio file in .wav format'
    )
    parser.add_argument(
        '--streams',
        type=int,
        default=20,
        help='The number of streams (audio sources)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='The number of workers for parallelism'
    )
    parser.add_argument(
        '--block-size',
        type=float,
        default=0.02,
        help='Size of signal blocks in seconds'
    )
    parser.add_argument(
        '--cut-audio-size',
        type=float,
        default=20 * 60,
        help='How many first seconds should be considered'
    )
    parser.add_argument(
        '--max-requests',
        type=int,
        default=1000000000
    )
    args = parser.parse_args()

    VoiceActivityDetector.DEVICE = torch.device('cpu')

    rate, signal, labels = load_labeled_audio(args.audio_path)

    X = int(rate * args.cut_audio_size)
    signal = signal[: X]
    labels = labels[: X]

    signal_size_f = len(signal)

    n_streams = args.streams
    n_workers = args.workers
    block_size_s = args.block_size
    block_size_f = int(block_size_s * rate)

    stream_signal_size_f = signal_size_f // n_streams
    stream_signals = []

    for i in range(n_streams):
        l = i * stream_signal_size_f
        r = l + stream_signal_size_f
        stream_signals.append(signal[l: r])

    queues = []
    for i in range(n_streams):
        q = Queue()
        queues.append(q)
    lock = Lock()

    delay_stories = []
    for i in range(len(queues)):
        delay_story = []
        delay_stories.append(delay_story)

    producer = Process(
        target=producer,
        args=(stream_signals, block_size_f, block_size_s, queues)
    )

    buffers = []
    buffer_locks = []
    for i, stream in enumerate(stream_signals):
        stream_buffer = StreamBuffer(rate)
        buffers.append(stream_buffer)
        buffer_lock = Lock()
        buffer_locks.append(buffer_lock)

    workers = []
    for i in range(n_workers):
        detector = VoiceActivityDetector()
        detector.load(args.model_path)
        detector.setup(rate)

        w = Process(
            target=worker,
            args=(lock, i, queues, buffers, detector, buffer_locks,
                  args.max_requests, delay_stories, args.block_size * 1000)
        )
        workers.append(w)

    producer.start()
    for w in workers:
        w.start()

    producer.join()
    for w in workers:
        w.join()

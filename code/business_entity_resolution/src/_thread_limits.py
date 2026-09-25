"""
Caps thread-pool size for numpy/BLAS/OpenMP-backed libraries (numpy, scipy,
sklearn, faiss) BEFORE they are ever imported anywhere in the process.

MUST be the first import in every entry-point script (train.py, infer.py,
blocking_report.py, benchmark_*.py, diagnose_tfidf.py) — these libraries
typically size their thread pool at import/first-use time, so setting the
env vars after numpy has already been imported elsewhere does nothing.

Why this exists: on gpu74 (a heavily shared multi-user box), letting these
libraries default to "use every core" causes catastrophic oversubscription.
Measured directly: the embedding channel's FAISS search step accumulated
CPU time at ~36-38x wall-clock rate (extreme thread contention/thrashing)
while its own tqdm progress counter stayed completely frozen for 90+
seconds — CPU was being burned on context-switching overhead between too
many competing threads, not on the actual computation making progress.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

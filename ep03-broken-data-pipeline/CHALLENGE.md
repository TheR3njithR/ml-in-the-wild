# Student challenge

After a change that looked harmless, one pipeline starts dropping an **extra 3% of rows every week**. It is a slow leak, not a shock: week 1 is only 3% short, week 4 is about 12% short, week 8 is about 22% short.

Using the code in this folder (you may add a generator file, but do not change the original seed or the locked episode numbers):

1. Plant this leak on one test-period pipeline, starting on a date you choose after 1 Aug 2025.
2. Run the existing expected-volume detector with the **same** amber and red cuts.
3. Tell me: does it catch the leak, and in which week does the daily ratio first cross amber? Which week, if any, does it cross red?
4. Add **one** feature that would have caught it earlier (an eight-week slope on usable_rows is a fair candidate). Show the new week it first alerts, and confirm you did not flood the board with new false alarms on the other pipelines.

Best answer gets pinned on the video.

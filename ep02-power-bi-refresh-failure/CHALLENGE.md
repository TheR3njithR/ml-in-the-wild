# Student challenge, Episode 2

1. In `make_data.py`, add a 9th incident type: every Sunday, one dataset silently falls back from incremental refresh to a full refresh, and then fails on memory halfway through.
2. Add several of these Sundays to the test months and run `./run_all.sh`.
3. Answer:
   - Does the model catch it from the features we already have?
   - Which feature moves first (`duration_trend`, `size_vs_28d`, `mem_over_90_min`, or something else)?
   - What ONE feature would you add to catch it earlier? (Hint: the dataset knows whether it planned an incremental refresh.)
4. Bonus: re-run `04_feedback_loop.py`. Does your new feature change the champion vs challenger decision?

Post your answer in the YouTube comments. The best answer gets pinned.

# Student challenge, Episode 1

1. In `make_data.py`, add an 8th incident type, `TIMEZONE_SHIFT`: on one day, 20% of one city's orders, revenue and refunds move into the next day (so one day is low and the next is high; sessions are unchanged).
2. Add two or three of them to the test year and run `./run_all.sh`.
3. Answer:
   - Does the model catch it? On which of the two days?
   - Which of the four features flags it (see the reason in the trust report from `03_decide_and_act.py <date>`)?
   - What ONE feature would you add to catch it better? (Hint: think about yesterday and today together.)
4. Bonus: re-run `04_feedback_loop.py`. Does your new feature change the champion vs challenger decision?

Post your answer in the YouTube comments. The best answer gets pinned.

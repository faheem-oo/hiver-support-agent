# AppleSupport AI Customer Support Agent

An AI customer-support pipeline built from the Customer Support on Twitter dataset. The system classifies incoming customer messages, retrieves relevant historical AppleSupport responses, drafts a grounded reply, and decides whether the issue should be escalated.

1. Problem Framing

Customer-support messages on Twitter are short, noisy, incomplete, and often contain multiple symptoms in one message.

The system therefore solves three connected tasks:

1. **Intent classification** — identify the primary customer-support intent.
2. **Historical response retrieval** — retrieve similar AppleSupport customer/support interactions.
3. **Escalation decision** — decide whether the issue can be handled automatically or requires additional support.

The selected brand is **AppleSupport** because it has a large number of customer-support interactions in the dataset and contains a broad range of technical support issues.

A key design principle is to avoid pretending that every historical response represents a successful resolution. Many conversations move to direct messages, so historical examples are treated as evidence rather than guaranteed solutions.

---

2. Dataset and Processing

Dataset: **Customer Support on Twitter (TWCS)**.

The raw dataset contains approximately 2.8 million tweets and seven fields:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

AppleSupport contains approximately **106,860 outbound support tweets** and approximately **106,623 inbound customer tweets mentioning AppleSupport** in the extracted data.

The pipeline:

```text
Raw TWCS dataset
       ↓
AppleSupport filtering
       ↓
Customer/support thread construction
       ↓
Text preprocessing
       ↓
Intent taxonomy
       ↓
Weak-label training data
       ↓
TF-IDF + Logistic Regression
       ↓
Historical response retrieval
       ↓
Reply drafting + escalation



3. Intent Taxonomy

Eleven intents were defined from the observed AppleSupport support traffic:

Intent	Description
ios_update	iOS/software updates and update-related problems
battery_charging	Battery drain, battery health, charging and power
device_performance	Slow, freezing, crashing, restarting or unresponsive devices
screen_display	Screen, touch, display and visual problems
connectivity	Wi-Fi, Bluetooth, cellular and connection problems
apple_id_icloud	Apple ID, iCloud, passwords and account access
apps	Problems with individual applications
itunes_music	Apple Music, iTunes, music library and playback
app_store_purchases	App Store, purchases, payments and refunds
apple_watch	Apple Watch-specific issues
other	Unclear, insufficient or uncovered requests

other is deliberately retained because forcing ambiguous Twitter messages into a specific technical category creates misleading confidence.

4. Training and Baselines
Baseline 1 — Majority Class

Every message is assigned the most common intent.

Accuracy: 34.0%
Macro F1: 4.61%
Baseline 2 — Keyword Rules

Hand-written keyword rules are used to identify the intent.

Accuracy: 68.5%
Macro F1: 66.03%
Final Intent Model

The final classifier uses:

Word-level TF-IDF
Unigrams and bigrams
Up to 50,000 features
Logistic Regression
Class-balanced training
Weakly labelled AppleSupport examples

The final model was evaluated on a frozen held-out set that was not used for model training.

5. Golden Set and Evaluation Design

A 200-example golden set was created and manually labelled with the defined intent taxonomy and escalation labels.

The golden set was split once using a fixed random seed:

150 examples — development set
50 examples — frozen held-out test set

The held-out test set was not used for model training or tuning.

This distinction is important because the model showed substantially higher performance on development data when those examples influenced training. The final headline numbers therefore use the untouched 50-example test set instead.

6. Final Held-Out Results
Intent Classification
Metric	Result
Accuracy	72.0%
Macro F1	66.66%
Errors	14 / 50
Escalation
Metric	Result
Accuracy	72.0%
Macro F1	65.28%
Escalation F1	50.0%
Errors	14 / 50

Escalation confusion matrix:

                 Predicted
              No       Yes
Actual No     29        5
Actual Yes     9        7

For the positive escalation class:

Precision: 58.3%
Recall: 43.8%
F1: 50.0%

The system is therefore more conservative than ideal about escalation recall: it misses some cases that should be escalated.

7. Retrieval and Reply Generation

Historical AppleSupport conversations are converted into customer-message → support-response pairs.

Approximately 103,614 historical pairs were generated.

Generic responses that only redirect the customer to direct messages are treated as weaker evidence.

Retrieval uses TF-IDF similarity and combines:

semantic similarity
historical response quality
topic overlap
penalty for generic DM-only responses

The final response is generated using intent-specific templates informed by the retrieved evidence.

The system does not claim that a historical troubleshooting step definitely solved the customer's problem.

When evidence is weak, the agent asks for additional information instead of inventing a solution.

8. Escalation Policy

Escalation is triggered by high-risk or unresolved signals such as:

account security or hacking concerns
data loss
hardware damage or repair needs
severe device failure
repeated restarting/freezing
repeated troubleshooting failures
account recovery problems
purchase/refund/financial issues
multiple critical problems
insufficient historical evidence

Example:

"My phone keeps resetting itself as I use it."

This should be treated as a repeated device failure rather than a normal troubleshooting request.

The policy intentionally prioritizes safety over aggressive automation for security, data-loss and severe-device cases.

9. Reply-Quality Evaluation

A 20-example human review was performed on held-out generated replies.

Each reply was scored from 1–5 on:

Dimension	Score
Relevance	2.65 / 5
Groundedness	4.20 / 5
Helpfulness	2.10 / 5
Specificity	2.20 / 5
Safety / escalation	3.45 / 5
Overall	2.92 / 5

Pass rate: 25%

The results show an important trade-off: the system is relatively grounded and cautious, but generated replies are often too generic to be highly helpful.

No LLM-judge score is reported unless an API-backed evaluation is actually run.

Human-agreement limitation

### Optional LLM-Based Reply Judge

An optional API-backed LLM judge can be used to evaluate generated replies on:

- relevance
- groundedness
- helpfulness
- specificity
- safety / escalation
- overall quality

The evaluator uses Gemini through the Google GenAI SDK.

The API key is intentionally not stored in the repository.

To enable the judge locally:

1. Create a Gemini API key using Google AI Studio.
2. Create a `.env` file in the project root.
3. Add:

```text
GEMINI_API_KEY=your_api_key_here

Install the required packages:
pip install google-genai python-dotenv
Run:
python src/evaluate_reply_quality.py

The evaluator writes results to:

outputs/llm_reply_quality.csv

LLM-judge results are only reported when the API-backed evaluation completes successfully. No scores are fabricated when the API is unavailable or quota-limited.

The current reply-quality review has one human evaluator. Therefore, it is not claimed as inter-annotator agreement and no Cohen's kappa is reported.

A second independent evaluator would be required to make a genuine agreement claim.

10. Top Five Failure Modes
1. Ambiguous multi-symptom messages

Example:

"Since iOS 11. Phone getting freeze so constant and battery is like never charged?"

The message contains an update, freezing and battery symptoms. A single-label classifier may select battery_charging even though the primary issue may be ios_update or device_performance.

Hypothesis: multi-intent classification would better represent these messages.

2. Update-related language dominates the actual symptom

Example:

"The Arabic/Urdu text is always cut off in the new iOS11 font."

The model predicted ios_update, while the gold label is screen_display.

Hypothesis: the classifier overweights high-frequency update/version vocabulary instead of identifying the actual customer symptom.

3. Very short or fragmented Twitter messages

Example:

"and fixing the individual songs by hand... IF you can FIND the songs."

The message contains insufficient context but was labelled itunes_music.

Hypothesis: conversation-level context would improve classification compared with classifying an isolated tweet.

4. Hardware/connectivity problems with indirect wording

Example:

"why does this keep happening to my Beats studio wireless? I've already had them replaced once for the same issue."

The model predicted other instead of connectivity.

Hypothesis: device-specific entities and repeated-failure context need stronger representation.

5. Generic replies reduce helpfulness

The human review produced only 2.10/5 for helpfulness and 2.20/5 for specificity.

The system often asks for device/version/error information instead of providing a concrete troubleshooting action.

Hypothesis: retrieval needs to prioritize successful or resolution-bearing historical examples rather than merely similar conversations.

11. What Is Misleading About My Headline Number?

The 72.0% accuracy should not be interpreted as "the support agent solves 72% of customer problems."

It measures classification correctness on only 50 frozen examples.

It also hides class imbalance: other has 16 of the 50 test examples, while some intents have only 1–3 examples.

For this reason, macro F1 (66.66%) is also reported.

Similarly, escalation accuracy does not mean that 72% of escalation decisions are safe or correct in production. The positive-class escalation recall is only 43.8%, meaning the current policy misses a substantial fraction of cases that the gold labels marked for escalation.

Reply quality is also materially weaker than classification performance: the human review produced an overall score of 2.92/5.

The headline therefore describes a useful prototype, not a production-ready autonomous support agent.

12. Engineering Decision Log
Selected AppleSupport because it provides a large and diverse support corpus.
Used conversation threads instead of isolated tweet pairs where possible.
Defined a small 11-intent taxonomy to keep classification operational.
Retained other rather than forcing uncertain messages into technical categories.
Used weak labels to obtain enough training data without manually labelling thousands of examples.
Created a 200-example golden set for evaluation.
Frozen 50 examples for final testing to reduce evaluation leakage.
Compared against a majority baseline to establish a minimum reference point.
Compared against keyword rules to test whether ML improves over simple heuristics.
Used class-balanced Logistic Regression because the intent distribution is highly uneven.
Used historical retrieval instead of generating unsupported technical solutions.
Penalized generic DM-only historical responses during retrieval.
Escalated security and severe device cases conservatively.
Did not report the V2 development score as the final result because the development examples influenced that experiment.
Reported human reply-quality limitations explicitly rather than fabricating LLM-judge or inter-annotator results.
13. One-Week Next Steps
Days 1–2: Better labels

Increase the golden set and have a second human independently label a subset.

Measure inter-annotator agreement and refine ambiguous intent definitions.

Days 3–4: Better classification

Test:

conversation-level context
multi-label intents
character n-grams
improved handling of update-related symptoms
confidence calibration
Day 5: Better retrieval

Prioritize historical responses that contain concrete troubleshooting steps or evidence of successful resolution.

Day 6: Better escalation

Optimize for escalation recall on security, data-loss, severe-device and repeated-failure cases.

Day 7: Evaluation

Run the same frozen test harness and compare:

accuracy
macro F1
escalation precision/recall/F1
reply-quality scores
failure categories

No test-set tuning should be performed.

14. Reproducibility

Create and activate a Python virtual environment:

python -m venv venv
source venv/Scripts/activate

Install dependencies:

pip install pandas numpy scikit-learn joblib google-genai python-dotenv
pip install google-genai python-dotenv

Run the main pipeline components:

python src/analyze_brand.py
python src/build_threads.py
python src/prepare_training_data.py
python src/create_weak_labels.py
python src/train_intent_model.py
python src/build_retrieval_data.py
python src/retrieve_responses.py

Create/evaluate the golden set:

python src/create_golden_set.py
python src/create_eval_split.py
python src/evaluate_baselines.py
python src/evaluate_heldout.py

Evaluate the full agent:

python src/evaluate_agent.py

Run the interactive agent:

python src/agent.py

The raw dataset is intentionally excluded from Git because of its size. Trained model binaries are also excluded from Git.

15. Repository Structure
hiver-support-agent/
│
├── data/
│   └── processed/
│       ├── applesupport_customer_messages.csv
│       ├── applesupport_threads.csv
│       ├── applesupport_retrieval.csv
│       ├── applesupport_training_messages.csv
│       ├── golden_set.csv
│       ├── golden_dev.csv
│       ├── golden_test.csv
│       └── weak_labels.csv
│
├── models/
│   ├── intent_classifier.joblib
│   ├── intent_tfidf.joblib
│   ├── retrieval_matrix.joblib
│   └── retrieval_tfidf.joblib
│
├── outputs/
│   ├── baseline_results.csv
│   ├── agent_eval.csv
│   ├── heldout_results.csv
│   ├── human_reply_quality.csv
│   └── intent_model_results.csv
│
├── src/
│   ├── agent.py
│   ├── analyze_brand.py
│   ├── build_threads.py
│   ├── build_retrieval_data.py
│   ├── create_eval_split.py
│   ├── create_golden_set.py
│   ├── create_human_eval.py
│   ├── create_weak_labels.py
│   ├── evaluate_agent.py
│   ├── evaluate_baselines.py
│   ├── evaluate_heldout.py
│   ├── evaluate_reply_quality.py
│   ├── retrieve_responses.py
│   └── train_intent_model.py
│
└── README.md
Final Assessment

This project demonstrates a complete support-agent prototype with:

reproducible preprocessing
explicit intent taxonomy
weakly supervised training
baseline comparison
frozen held-out evaluation
historical response retrieval
deterministic reply drafting
escalation policy
human reply-quality evaluation
failure analysis
documented engineering decisions

The current system is best viewed as a measurable prototype, not a production autonomous support system. The evaluation shows that classification is promising but ambiguous multi-symptom messages, retrieval quality, and escalation recall remain the main areas for improvement.
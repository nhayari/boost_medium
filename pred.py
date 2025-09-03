import pandas as pd
from medium.params import *
from medium.ml_logic.registry import *
import jsonlines

records = []
line_count = 0
with jsonlines.open('raw_data/test.json', mode='r') as reader:
            for obj in reader:
                records.append(obj)
                line_count += 1
                # Stop if we've reached the requested number of lines
print(f"Loaded {len(records)} lines from {'raw_data/test.json'}")

df = pd.DataFrame(records)

model = load_model('Ridge_punct_removed_stopwords_removed_data_scaled')
df['log1p_recommends'] = 0

y_pred = model.predict(df) # type: ignore

y_pred = pd.Series(y_pred)

y_pred.to_csv('predictions.csv')

submission_template = pd.read_csv('raw_data/sample_submission.csv')
submission_ids = submission_template['id']

submission = pd.concat([submission_ids, y_pred], axis=1, ignore_index=True)
submission.to_csv('submission.csv')

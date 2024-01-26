# Comparing Methods for Identifying Novel Indications of Drugs using Electronic Health Records

Computational drug re-purposing has received a lot of attention in the past decade. However, methods developed to date focused on established compounds for which information on both, successfully treated patients and chemical and genomic impact, were known. Such information does not always exist for first-in-class drugs under development. To identify indications (diseases) for drugs under development we extended and tested several unsupervised computational methods that utilize Electronic Health Record (EHR) data. We tested the methods on known drugs with multiple indications and show that a variant of matrix factorization leads to the best performance for first-in-line drugs.

# Install
```
python3 -m venv indication_example
source indication_example/bin/activate
python3 -m pip install -e .
```

# Login
```
source indication_example/bin/activate
```

# Training matrix factorization
```
spark-submit --master [local[2],or other config] src/train_models/mf_factory.py --input_data s3a://path/to-your-data
```

# Input data format
The data is assumed to be stored on s3 in parquet files with the following schema: person_id (int), feature_name (string). 
The feature names are the events that happened to each person in their medical records.

# Training config
The training config is set (and can be changed) in config/training_mf.yaml

# Output data format
The output is saved in the output_location, and contains embeddings for all features, including drug indications.

## Contacts
For any inquiries please raise a git issue and we will try to follow-up in a timely manner.

# License
License Notice: Permission is hereby granted, free of charge, for academic research purposes only and for non-commercial uses only, to any person from academic research or non-profit organizations obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, or merge the Software, subject to the following conditions: this permission notice shall be included in all copies of the Software or of substantial portions of the Software. For purposes of this license, “non-commercial use” excludes uses foreseeably resulting in a commercial benefit. All other rights are reserved. The Software is provided “as is”, without warranty of any kind, express or implied, including the warranties of noninfringement.

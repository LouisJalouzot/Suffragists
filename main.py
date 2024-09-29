from src.preprocess import preprocess_pdfs

preprocess_pdfs("data/test/*.pdf", "preprocessed/test", n_jobs=8)

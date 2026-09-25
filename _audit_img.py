import pandas as pd

for f in ('results_images/image_results.csv',
          'results_images_cifar_all/image_results.csv',
          'results_images_cifar/image_results.csv'):
    try:
        df = pd.read_csv(f)
        print(f, '| rows:', len(df), '| datasets:', sorted(df.dataset.unique()),
              '| alphas:', sorted(df.alpha.unique().tolist())[:8])
    except Exception as e:
        print(f, '-> ERROR:', type(e).__name__, str(e)[:100])

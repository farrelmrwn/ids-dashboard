from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import uuid
from ml_pipeline import run_pipeline

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    table_html = None
    download_path = None
    chart_data = None
    metrics_table = None

    if request.method == 'POST':
        file = request.files['file']
        dataset = request.form.get('dataset') or 'unsw'

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.csv")
            file.save(filepath)

            # Jalankan pipeline
            output_df, model_metrics = run_pipeline(filepath, dataset=dataset)

            # Simpan hasil prediksi
            result_path = filepath.replace('.csv', '_predicted.csv')
            output_df.to_csv(result_path, index=False)

            # Tampilkan tabel hasil (50 baris pertama)
            table_html = output_df.head(50).to_html(classes='data')

            # Chart Data (Gunakan macro average f1 jika tersedia)
            chart_data = {
                'labels': list(model_metrics.keys()),
                'datasets': [
                    {
                        'label': 'Accuracy (%)',
                        'data': [round(m.get('accuracy', 0), 2) for m in model_metrics.values()],
                        'backgroundColor': 'rgba(75, 192, 192, 0.6)'
                    },
                    {
                        'label': 'F1 Score Macro Avg (%)',
                        'data': [round(m.get('macro_avg_f1', 0), 2) for m in model_metrics.values()],
                        'backgroundColor': 'rgba(153, 102, 255, 0.6)'
                    },
                    {
                        'label': 'Latency (ms)',
                        'data': [round(m.get('latency', 0), 2) for m in model_metrics.values()],
                        'backgroundColor': 'rgba(255, 99, 132, 0.6)'
                    }
                ]
            }

            # Ubah ke DataFrame tabel metrik
            metrics_df = pd.DataFrame(model_metrics).T.reset_index()
            metrics_df.rename(columns={'index': 'Model'}, inplace=True)
            metrics_table = metrics_df.to_html(classes='table table-bordered', index=False)

            return render_template(
                'index.html',
                tables=table_html,
                metrics_table=metrics_table,
                download_link=result_path,
                chart_data=chart_data
            )

    return render_template('index.html')

@app.route('/download')
def download():
    path = request.args.get('path')
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
document.getElementById('upload-form').addEventListener('submit', function(e) {
});

document.getElementById('predict-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const queryText = document.getElementById('query').value.trim();
  const resultsDiv = document.getElementById('predict-results');
  const docsDiv = document.getElementById('docs-results');
  const canvas = document.getElementById('simChart');
  let simChart = Chart.getChart("simChart");

  if (!queryText) {
    resultsDiv.innerHTML = '';
    docsDiv.innerHTML = '';
    if (simChart) simChart.destroy();
    return;
  }

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: queryText })
  })
  .then(response => response.json())
  .then(data => {
    let htmlSug = '<h3>Prediksi Kata Berikutnya</h3>';
    if (data.suggestions.length > 0) {
      htmlSug += '<table><thead><tr><th>Kata Berikutnya</th><th>Probabilitas</th></tr></thead><tbody>';
      data.suggestions.forEach(item => {
        htmlSug += `<tr><td>${item.word}</td><td>${item.prob.toFixed(4)}</td></tr>`;
      });
      htmlSug += '</tbody></table>';
    } else {
      htmlSug += `<p><em>Tidak ada saran untuk "<strong>${queryText}</strong>".</em></p>`;
    }
    resultsDiv.innerHTML = htmlSug;

    let htmlDocs = '<h3>Dokumen Relevan</h3>';
    const labels = [];
    const scores = [];
    if (data.docs.length > 0) {
      htmlDocs += '<table><thead><tr><th>Nama Dokumen</th><th>Skor Relevansi</th></tr></thead><tbody>';
      data.docs.forEach(doc => {
        labels.push(doc.filename);
        scores.push(doc.score);
        htmlDocs += `<tr><td>${doc.filename}</td><td>${doc.score.toFixed(4)}</td></tr>`;
      });
      htmlDocs += '</tbody></table>';
    } else {
      htmlDocs += `<p><em>Tidak ada dokumen relevan untuk "<strong>${queryText}</strong>".</em></p>`;
    }
    docsDiv.innerHTML = htmlDocs;

    if (simChart) simChart.destroy();

    if (labels.length > 0) {
      const ctx = canvas.getContext('2d');
      simChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Skor Cosine Similarity',
            data: scores,
            borderWidth: 1,
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)'
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true,
              title: { display: true, text: 'Similarity' }
            },
            x: {
              title: { display: true, text: 'Dokumen' }
            }
          },
          plugins: {
            legend: { display: false }
          }
        }
      });
    }
  })
  .catch(err => {
    console.error('Error pada API predict:', err);
    resultsDiv.innerHTML = '<p><em>Terjadi kesalahan saat prediksi.</em></p>';
    docsDiv.innerHTML = '<p><em>Terjadi kesalahan saat mengambil dokumen.</em></p>';
    if (simChart) simChart.destroy();
  });
});

document.getElementById('upload-form').addEventListener('submit', e=>{});

document.getElementById('predict-form').addEventListener('submit', function(e) {
  e.preventDefault();
  const q = document.getElementById('query').value.trim();
  const outSug = document.getElementById('predict-results');
  const outDocs = document.getElementById('docs-results');
  const canvas = document.getElementById('simChart');
  let simChart = Chart.getChart("simChart");

  if (!q) {
    outSug.innerHTML = '';
    outDocs.innerHTML = '';
    if (simChart) simChart.destroy();
    return;
  }

  outSug.innerHTML = '<div class="spinner"></div>';
  outDocs.innerHTML = '';

  fetch('/api/predict', {
    method:  'POST',
    headers: {'Content-Type':'application/json'},
    body:    JSON.stringify({query:q})
  })
  .then(r => r.json())
  .then(data => {
    let hs = '<h3>Prediksi Kata Berikutnya</h3>';
    if (data.suggestions.length) {
      hs += '<table><thead><tr><th>Kata</th><th>Prob.</th></tr></thead><tbody>';
      data.suggestions.forEach(s => {
        hs += `<tr><td>${s.word}</td><td>${s.prob.toFixed(4)}</td></tr>`;
      });
      hs += '</tbody></table>';
    } else {
      hs += `<p class="empty">Tidak ada saran untuk "<strong>${q}</strong>".</p>`;
    }
    outSug.innerHTML = hs;

    let hd = '<h3>Dokumen Relevan</h3>';
    const labels = [], scores = [];
    if (data.docs.length) {
      hd += '<table><thead><tr><th>Dokumen</th><th>Skor</th><th>Preview</th></tr></thead><tbody>';
      data.docs.forEach(d => {
        labels.push(d.filename);
        scores.push(d.score);
        hd += `<tr>
                 <td>${d.filename}</td>
                 <td>${d.score.toFixed(4)}</td>
                 <td class="preview">${d.snippet}</td>
               </tr>`;
      });
      hd += '</tbody></table>';
    } else {
      hd += `<p class="empty">Tidak ada dokumen relevan untuk "<strong>${q}</strong>".</p>`;
    }
    outDocs.innerHTML = hd;

    // chart
    if (simChart) simChart.destroy();
    if (labels.length) {
      const ctx = canvas.getContext('2d');
      simChart = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [{
          label:'Skor Cosine Similarity',
          data:scores,
          backgroundColor:'rgba(30,58,138,0.6)',
          borderColor:'rgba(30,58,138,1)',
          borderWidth:1
        }]},
        options: {
          responsive:true,
          scales:{
            y:{beginAtZero:true, title:{display:true,text:'Similarity'}},
            x:{ title:{display:true,text:'Dokumen'} }
          },
          plugins:{legend:{display:false}}
        }
      });
    }
  })
  .catch(err=>{
    console.error(err);
    outSug.innerHTML = '<p class="empty">Terjadi kesalahan.</p>';
    outDocs.innerHTML= '<p class="empty">Terjadi kesalahan.</p>';
    if (simChart) simChart.destroy();
  });
});

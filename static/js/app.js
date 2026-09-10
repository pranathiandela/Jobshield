document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("detector-form");
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const result = document.getElementById("result");
    result.hidden = false;
    result.innerHTML = '<div class="notice">Scanning listing… checking language, contact signals and compensation claims.</div>';
    const payload = {
      job_text: document.getElementById("job_text").value,
      job_url: document.getElementById("job_url").value,
      recruiter_email: document.getElementById("recruiter_email").value
    };
    try {
      const r = await fetch("/api/analyze-job", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await r.json();
      const signals = data.signals.map(s => `<div class="signal">${escapeHtml(s)}</div>`).join("");
      result.innerHTML = `<div class="grid-2">
        <div><div class="small">RISK SCORE</div><div class="score">${data.score}</div><div class="risk">${escapeHtml(data.risk)} RISK</div><p style="color:var(--silver-mist)">${escapeHtml(data.summary)}</p></div>
        <div><div class="small">SUSPICIOUS SIGNALS</div><div class="signal-list">${signals || '<div class="signal">No strong rule-based red flags detected.</div>'}</div></div>
      </div>`;
    } catch (err) {
      result.innerHTML = '<div class="notice">The scan could not be completed. Make sure Flask is running.</div>';
    }
  });
});
function escapeHtml(v){return String(v).replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]));}

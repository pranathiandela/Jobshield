document.addEventListener("DOMContentLoaded", () => {

  const form = document.getElementById("detector-form");

  if (!form) return;


  const result =
    document.getElementById("result");

  const progress =
    document.getElementById("scan-progress");

  const fill =
    document.getElementById("progress-fill");

  const label =
    document.getElementById("progress-label");

  const button =
    document.getElementById("scan-btn");


  form.addEventListener("submit", async (e) => {

    e.preventDefault();


    const descriptionElement =
      document.getElementById("job_text");

    const screenshotElement =
      document.getElementById("screenshot");


    const description =
      descriptionElement
        ? descriptionElement.value.trim()
        : "";


    const screenshotSelected =
      screenshotElement &&
      screenshotElement.files &&
      screenshotElement.files.length > 0;


    /*
     * User must provide either:
     *
     * 1. Job description
     * OR
     * 2. Screenshot
     */

    if (!description && !screenshotSelected) {

      result.hidden = false;

      result.innerHTML =
        '<div class="notice">' +
        'Please enter a job description or upload a screenshot before scanning.' +
        '</div>';


      if (descriptionElement) {
        descriptionElement.focus();
      }

      return;
    }


    /*
     * Disable button
     */

    button.disabled = true;

    button.style.opacity = ".65";


    /*
     * Show progress
     */

    progress.hidden = false;

    result.hidden = true;


    fill.style.width = "0%";

    label.textContent =
      "Preparing verification...";


    /*
     * Scanning steps
     */

    const scanSteps = [

      [15, "Extracting job and company details..."],

      [32, "Checking company and domain signals..."],

      [48, "Checking careers and job consistency..."],

      [64, "Checking recruiter and contact signals..."],

      [80, "Analyzing salary and scam patterns..."],

      [94, "Collecting evidence and calculating score..."]

    ];


    try {

      for (const step of scanSteps) {

        await wait(220);

        fill.style.width =
          step[0] + "%";

        label.textContent =
          step[1];

      }


      /*
       * Create form data
       */

      const payload =
        new FormData(form);


      /*
       * Send to Flask backend
       */

      const response =
        await fetch("/api/analyze-job", {

          method: "POST",

          body: payload

        });


      let data;


      try {

        data =
          await response.json();

      } catch (jsonError) {

        throw new Error(
          "The server returned an invalid response."
        );

      }


      if (!response.ok) {

        throw new Error(
          data.error || "Scan failed."
        );

      }


      /*
       * Complete progress
       */

      fill.style.width = "100%";

      label.textContent =
        "Verification complete.";


      await wait(300);


      /*
       * Render result
       */

      renderResult(data);


    } catch (error) {

      console.error(
        "JobShield verification error:",
        error
      );


      result.hidden = false;


      result.innerHTML =
        '<div class="notice">' +
        escapeHtml(
          error.message ||
          "The scan could not be completed."
        ) +
        "</div>";


    } finally {

      button.disabled = false;

      button.style.opacity = "1";

    }

  });


  /*
   * ==========================================================
   * RENDER RESULT
   * ==========================================================
   */

  function renderResult(data) {

    const groups =
      data.checks || {};


    const evidence =
      data.evidence || [];


    const signals =
      data.suspicious_signals || [];


    const safe =
      data.safe_signals || [];


    const nextSteps =
      data.next_steps || [];


    const score =
      Number(
        data.score ??
        data.legitimacy_score ??
        0
      );


    const verdict =
      data.verdict ||
      data.risk ||
      "Caution";


    /*
     * Get the correct visual class
     *
     * SAFE   -> safe
     * RISKY  -> risky
     * MEDIUM -> medium
     */

    const verdictClass =
      getVerdictClass(
        verdict,
        score
      );


    const checkRows =
      Object.entries(groups)

        .map(([key, value]) => {

          const item =
            typeof value === "object" &&
            value !== null

              ? value

              : {
                  status: value
                };


          return `

            <div class="metric">

              <span>
                ${escapeHtml(
                  formatKey(key)
                )}
              </span>

              <strong>
                ${escapeHtml(
                  item.status ||
                  "Reviewed"
                )}
              </strong>

            </div>

          `;

        })

        .join("");


    const list =
      (items, className) => {

        if (
          !Array.isArray(items) ||
          items.length === 0
        ) {

          return `

            <div class="neutral">
              No items reported.
            </div>

          `;

        }


        return items

          .map((item) => {

            return `

              <div class="${className}">
                ${escapeHtml(item)}
              </div>

            `;

          })

          .join("");

      };


    /*
     * Result HTML
     */

    result.hidden = false;


    result.innerHTML = `

      <div class="result-hero">

        <div>

          <div class="result-score">

            <span id="animated-score">
              0
            </span>

            <small>
              LEGITIMACY / 100
            </small>

          </div>

        </div>


        <div>

          <div class="verdict ${verdictClass}">
            ${escapeHtml(verdict)}
          </div>


          <h2>

            ${escapeHtml(
              data.summary_title ||
              "Verification report"
            )}

          </h2>


          <p>

            ${escapeHtml(
              data.summary ||
              "Review the evidence below before applying or sharing sensitive information."
            )}

          </p>

        </div>

      </div>


      <div class="report-grid">


        <div class="report-card">

          <h3>
            Verification checks
          </h3>

          ${
            checkRows ||
            '<div class="neutral">Checks completed.</div>'
          }

        </div>


        <div class="report-card">

          <h3>
            Suspicious signals
          </h3>

          ${list(
            signals,
            "warn"
          )}

        </div>


        <div class="report-card">

          <h3>
            Safe signals
          </h3>

          ${list(
            safe,
            "good"
          )}

        </div>


        <div class="report-card">

          <h3>
            Evidence
          </h3>

          ${list(
            evidence,
            "neutral"
          )}

        </div>

      </div>


      <div class="report-card next-steps">

        <h3>
          Recommended next steps
        </h3>


        <ul>

          ${
            nextSteps.length

              ? nextSteps

                  .map((step) => {

                    return `

                      <li>
                        ${escapeHtml(step)}
                      </li>

                    `;

                  })

                  .join("")

              : `

                  <li>
                    Review the job and company details carefully before applying.
                  </li>

                `
          }

        </ul>

      </div>


      <div class="report-footer">

        <span class="small">
          Scan saved to your JobShield history.
        </span>


        <button
          class="btn btn-nav"
          type="button"
          id="print-report"
        >
          Print / Save report
        </button>

      </div>

    `;


    /*
     * Score animation
     */

    animateScore(score);


    /*
     * Print button
     */

    const printButton =
      document.getElementById(
        "print-report"
      );


    if (printButton) {

      printButton.addEventListener(
        "click",
        () => window.print()
      );

    }


    /*
     * Scroll to result
     */

    result.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  }


  /*
   * ==========================================================
   * VERDICT COLOR CLASS
   * ==========================================================
   */

  function getVerdictClass(verdict, score) {

    const value =
      String(verdict || "")
        .toLowerCase()
        .trim();


    /*
     * SAFE
     */

    if (
      value.includes("safe") ||
      value.includes("legitimate") ||
      value.includes("low")
    ) {

      return "safe";

    }


    /*
     * RISKY
     */

    if (
      value.includes("risky") ||
      value.includes("risk") ||
      value.includes("unsafe") ||
      value.includes("scam") ||
      value.includes("high")
    ) {

      return "risky";

    }


    /*
     * MEDIUM / CAUTION
     */

    if (
      value.includes("medium") ||
      value.includes("caution") ||
      value.includes("moderate")
    ) {

      return "medium";

    }


    /*
     * Fallback based on score
     */

    const numericScore =
      Number(score);


    if (numericScore >= 70) {

      return "safe";

    }


    if (numericScore >= 40) {

      return "medium";

    }


    return "risky";

  }


  /*
   * ==========================================================
   * SCORE ANIMATION
   * ==========================================================
   */

  function animateScore(targetScore) {

    const scoreElement =
      document.getElementById(
        "animated-score"
      );


    if (!scoreElement) return;


    targetScore =
      Number(targetScore);


    if (!Number.isFinite(targetScore)) {

      targetScore = 0;

    }


    targetScore =
      Math.max(
        0,
        Math.min(
          100,
          Math.round(targetScore)
        )
      );


    let currentScore = 0;


    scoreElement.textContent =
      "0";


    if (targetScore === 0) {
      return;
    }


    const animationSpeed = 30;


    const animation =
      setInterval(() => {

        currentScore++;


        scoreElement.textContent =
          String(currentScore);


        if (
          currentScore >=
          targetScore
        ) {

          clearInterval(animation);


          scoreElement.textContent =
            String(targetScore);

        }

      }, animationSpeed);

  }


  /*
   * ==========================================================
   * WAIT
   * ==========================================================
   */

  function wait(milliseconds) {

    return new Promise(
      (resolve) => {

        setTimeout(
          resolve,
          milliseconds
        );

      }
    );

  }


  /*
   * ==========================================================
   * FORMAT KEY
   * ==========================================================
   */

  function formatKey(value) {

    return String(value)

      .replace(
        /_/g,
        " "
      )

      .replace(
        /\b\w/g,
        (character) => {

          return character.toUpperCase();

        }
      );

  }


  /*
   * ==========================================================
   * ESCAPE HTML
   * ==========================================================
   */

  function escapeHtml(value) {

    return String(value ?? "")

      .replace(
        /[&<>"']/g,
        (character) => {

          const entities = {

            "&": "&amp;",

            "<": "&lt;",

            ">": "&gt;",

            '"': "&quot;",

            "'": "&#039;"

          };


          return entities[character];

        }
      );

  }

});
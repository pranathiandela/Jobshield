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


  /*
   * ==========================================================
   * DETECTOR INPUTS
   * ==========================================================
   */

  const descriptionElement =
    document.getElementById("job_text");

  const jobFileElement =
    document.getElementById("job_file");

  const jobUrlElement =
    document.getElementById("job_url");


  /*
   * ==========================================================
   * SUBMIT FORM
   * ==========================================================
   */

  form.addEventListener("submit", async (e) => {

    e.preventDefault();


    /*
     * Read current values
     */

    const description =
      descriptionElement
        ? descriptionElement.value.trim()
        : "";


    const jobFileSelected =
      jobFileElement &&
      jobFileElement.files &&
      jobFileElement.files.length > 0;


    const jobUrl =
      jobUrlElement
        ? jobUrlElement.value.trim()
        : "";


    /*
     * ========================================================
     * DETERMINE INPUT MODE
     * ========================================================
     */

    let inputMode = "";


    if (jobUrl) {

      inputMode = "url";

    } else if (jobFileSelected) {

      inputMode = "file";

    } else if (description) {

      inputMode = "description";

    }


    /*
     * ========================================================
     * VALIDATION
     * ========================================================
     */

    if (!inputMode) {

      result.hidden = false;

      result.innerHTML =
        '<div class="notice">' +
        'Please enter a job description, upload a PDF/DOCX/TXT file, or enter a job URL before scanning.' +
        '</div>';


      if (descriptionElement) {

        descriptionElement.focus();

      }

      return;

    }


    /*
     * URL validation
     */

    if (inputMode === "url") {

      let validUrl = false;

      try {

        const parsedUrl =
          new URL(jobUrl);

        validUrl =
          parsedUrl.protocol === "http:" ||
          parsedUrl.protocol === "https:";

      } catch (error) {

        validUrl = false;

      }


      if (!validUrl) {

        result.hidden = false;

        result.innerHTML =
          '<div class="notice">' +
          'Please enter a valid HTTP or HTTPS job URL.' +
          '</div>';


        if (jobUrlElement) {

          jobUrlElement.focus();

        }

        return;

      }

    }


    /*
     * ========================================================
     * DISABLE BUTTON
     * ========================================================
     */

    button.disabled = true;

    button.style.opacity = ".65";


    /*
     * ========================================================
     * SHOW PROGRESS
     * ========================================================
     */

    progress.hidden = false;

    result.hidden = true;


    fill.style.width = "0%";

    label.textContent =
      "Preparing verification...";


    /*
     * ========================================================
     * SCANNING STEPS
     * ========================================================
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
       * ========================================================
       * CREATE FORM DATA
       * ========================================================
       */

      const payload =
        new FormData(form);


      /*
       * Tell Flask which detector mode was selected.
       */

      payload.set(
        "input_mode",
        inputMode
      );


      /*
       * ========================================================
       * CLEAN INACTIVE INPUTS
       * ========================================================
       */

      if (inputMode === "url") {

        payload.delete("job_text");

        payload.delete("job_file");

      }


      if (inputMode === "description") {

        payload.delete("job_file");

      }


      if (inputMode === "file") {

        payload.delete("job_text");

      }


      /*
       * ========================================================
       * SEND TO FLASK BACKEND
       * ========================================================
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


      /*
       * ========================================================
       * SERVER ERROR
       * ========================================================
       */

      if (!response.ok) {

        throw new Error(
          data.error || "Scan failed."
        );

      }


      /*
       * ========================================================
       * COMPLETE PROGRESS
       * ========================================================
       */

      fill.style.width = "100%";

      label.textContent =
        "Verification complete.";


      await wait(300);


      /*
       * ========================================================
       * RENDER RESULT
       * ========================================================
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


    /*
     * ========================================================
     * LEGITIMACY SCORE
     * ========================================================
     */

    const rawScore =
      Number(
        data.score ??
        data.legitimacy_score ??
        0
      );


    const score =
      clamp(
        Number.isFinite(rawScore)
          ? Math.round(rawScore)
          : 0,
        0,
        100
      );


    const verdict =
      data.verdict ||
      data.risk ||
      "Caution";


    /*
     * ========================================================
     * ML ASSESSMENT
     * ========================================================
     */

    const mlAvailable =
      data.ml_available === true;


    const mlPrediction =
      data.ml_prediction ||
      "Unavailable";


    const rawMlPercentage =
      Number(
        data.ml_fraud_percentage
      );


    const mlFraudPercentage =
      Number.isFinite(rawMlPercentage)
        ? clamp(rawMlPercentage, 0, 100)
        : null;


    /*
     * ========================================================
     * VISUAL STATE
     * ========================================================
     */

    const verdictClass =
      getVerdictClass(
        verdict,
        score
      );


    /*
     * ========================================================
     * CHECK ROWS
     * ========================================================
     */

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


    /*
     * ========================================================
     * LIST RENDERER
     * ========================================================
     */

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
     * ========================================================
     * CIRCULAR SCORE
     *
     * Uses SVG so the visual does not depend on extra CSS.
     * The circle represents the REAL backend score.
     * ========================================================
     */

    const circleRadius = 52;

    const circleCircumference =
      2 * Math.PI * circleRadius;


    const scoreOffset =
      circleCircumference -
      (
        score / 100
      ) * circleCircumference;


    /*
     * ========================================================
     * ML BAR
     * ========================================================
     */

    const mlBarWidth =
      mlFraudPercentage !== null
        ? mlFraudPercentage
        : 0;


    /*
     * ========================================================
     * RESULT HTML
     * ========================================================
     */

    result.hidden = false;


    result.innerHTML = `

      <!-- ====================================================
           RESULT HERO
      ===================================================== -->

      <div class="result-hero">


        <!-- CIRCULAR SCORE -->

        <div
          class="score-ring"
          aria-label="Legitimacy score ${score} out of 100"
          style="
            width:154px;
            height:154px;
            position:relative;
            display:flex;
            align-items:center;
            justify-content:center;
            flex-shrink:0;
          "
        >

          <svg
            width="154"
            height="154"
            viewBox="0 0 154 154"
            style="
              position:absolute;
              inset:0;
              transform:rotate(-90deg);
            "
            aria-hidden="true"
          >

            <!-- Background circle -->

            <circle
              cx="77"
              cy="77"
              r="${circleRadius}"
              fill="none"
              stroke="rgba(255,255,255,.09)"
              stroke-width="9"
            ></circle>


            <!-- Score circle -->

            <circle
              id="score-circle"
              cx="77"
              cy="77"
              r="${circleRadius}"
              fill="none"
              stroke="currentColor"
              stroke-width="9"
              stroke-linecap="round"
              stroke-dasharray="${circleCircumference}"
              stroke-dashoffset="${circleCircumference}"
              style="
                transition:
                  stroke-dashoffset 1.2s ease;
              "
            ></circle>

          </svg>


          <div
            style="
              position:relative;
              z-index:2;
              text-align:center;
            "
          >

            <div
              id="animated-score"
              style="
                font-size:42px;
                font-weight:700;
                line-height:1;
                letter-spacing:-.04em;
              "
            >
              0
            </div>


            <div
              style="
                margin-top:7px;
                font-size:9px;
                letter-spacing:.14em;
                opacity:.65;
                white-space:nowrap;
              "
            >
              LEGITIMACY
            </div>

          </div>

        </div>


        <!-- RESULT SUMMARY -->

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


          <!-- SCORE DESCRIPTION -->

          <div
            style="
              margin-top:14px;
              font-size:12px;
              opacity:.68;
            "
          >

            Legitimacy score:
            <strong>
              ${score}/100
            </strong>

          </div>

        </div>

      </div>


      <!-- ====================================================
           REPORT GRID
      ===================================================== -->

      <div class="report-grid">


        <!-- VERIFICATION CHECKS -->

        <div class="report-card">

          <h3>
            Verification checks
          </h3>

          ${
            checkRows ||
            '<div class="neutral">Checks completed.</div>'
          }

        </div>


        <!-- SUSPICIOUS SIGNALS -->

        <div class="report-card">

          <h3>
            Suspicious signals
          </h3>

          ${list(
            signals,
            "warn"
          )}

        </div>


        <!-- SAFE SIGNALS -->

        <div class="report-card">

          <h3>
            Safe signals
          </h3>

          ${list(
            safe,
            "good"
          )}

        </div>


        <!-- EVIDENCE -->

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


      <!-- ====================================================
           ML ASSESSMENT
      ===================================================== -->

      ${
        mlAvailable
          ? `

            <div class="report-card ml-assessment">

              <h3>
                AI / ML Assessment
              </h3>


              <!-- FRAUD PROBABILITY -->

              <div
                style="
                  margin:4px 0 18px;
                "
              >

                <div
                  style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-bottom:9px;
                    font-size:13px;
                  "
                >

                  <span>
                    Fraud probability
                  </span>

                  <strong>
                    ${
                      mlFraudPercentage !== null
                        ? mlFraudPercentage.toFixed(2)
                        : "N/A"
                    }%
                  </strong>

                </div>


                <div
                  style="
                    width:100%;
                    height:9px;
                    overflow:hidden;
                    border-radius:999px;
                    background:rgba(255,255,255,.08);
                  "
                >

                  <div
                    id="ml-fraud-bar"
                    style="
                      width:0%;
                      height:100%;
                      border-radius:999px;
                      background:currentColor;
                      transition:width 1.1s ease;
                    "
                  ></div>

                </div>

              </div>


              <!-- ML PREDICTION -->

              <div class="metric">

                <span>
                  ML prediction
                </span>

                <strong>
                  ${escapeHtml(mlPrediction)}
                </strong>

              </div>


              <p class="neutral">

                This machine-learning signal is based on patterns
                learned from the training dataset. It is one screening
                signal and does not independently prove whether a job
                is fraudulent.

              </p>

            </div>

          `
          : ""
      }


      <!-- ====================================================
           NEXT STEPS
      ===================================================== -->

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


      <!-- ====================================================
           FOOTER
      ===================================================== -->

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
     * ========================================================
     * ANIMATE SCORE
     * ========================================================
     */

    animateScore(score);


    /*
     * ========================================================
     * ANIMATE SCORE CIRCLE
     * ========================================================
     */

    const scoreCircle =
      document.getElementById(
        "score-circle"
      );


    if (scoreCircle) {

      /*
       * Start completely empty.
       */

      scoreCircle.style.strokeDashoffset =
        String(circleCircumference);


      /*
       * Small delay makes the animation visible.
       */

      requestAnimationFrame(() => {

        requestAnimationFrame(() => {

          scoreCircle.style.strokeDashoffset =
            String(scoreOffset);

        });

      });

    }


    /*
     * ========================================================
     * ML FRAUD BAR
     * ========================================================
     */

    const mlBar =
      document.getElementById(
        "ml-fraud-bar"
      );


    if (
      mlBar &&
      mlFraudPercentage !== null
    ) {

      /*
       * Start empty.
       */

      mlBar.style.width =
        "0%";


      /*
       * Animate to the REAL backend value.
       */

      requestAnimationFrame(() => {

        requestAnimationFrame(() => {

          mlBar.style.width =
            mlBarWidth + "%";

        });

      });

    }


    /*
     * ========================================================
     * PRINT BUTTON
     * ========================================================
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
     * ========================================================
     * SCROLL TO RESULT
     * ========================================================
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
     * FALLBACK BASED ON SCORE
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


    const animationSpeed = 18;


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
   * CLAMP NUMBER
   * ==========================================================
   */

  function clamp(value, minimum, maximum) {

    return Math.min(
      maximum,
      Math.max(
        minimum,
        value
      )
    );

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
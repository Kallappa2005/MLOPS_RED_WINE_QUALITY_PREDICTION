document.addEventListener("DOMContentLoaded", function () {
    // -----------------------------------------------------------------------
    // Authentication & Secure Fetch Helper
    // -----------------------------------------------------------------------
    const loginOverlay = document.getElementById("login-overlay");
    const loginForm = document.getElementById("login-form");
    const loginError = document.getElementById("login-error");
    const btnLogout = document.getElementById("btn-logout");
    const userIdentityBadge = document.getElementById("user-identity-badge");

    function checkAuth() {
        const token = localStorage.getItem("jwt_token");
        const role = localStorage.getItem("user_role");
        const username = localStorage.getItem("user_name");

        if (!token) {
            loginOverlay.style.display = "flex";
            return false;
        } else {
            loginOverlay.style.display = "none";
            if (userIdentityBadge) {
                userIdentityBadge.textContent = `${username.toUpperCase()} (${role})`;
            }
            return true;
        }
    }

    if (loginForm) {
        loginForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const username = document.getElementById("login-username").value;
            const password = document.getElementById("login-password").value;

            fetch("/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password })
            })
            .then(res => {
                if (!res.ok) {
                    return res.json().then(err => { throw new Error(err.error || "Authentication failed") });
                }
                return res.json();
            })
            .then(data => {
                localStorage.setItem("jwt_token", data.token);
                localStorage.setItem("user_role", data.role);
                localStorage.setItem("user_name", data.username);
                loginError.style.display = "none";
                checkAuth();
                loadTabContent("xai");
            })
            .catch(err => {
                loginError.textContent = err.message;
                loginError.style.display = "block";
            });
        });
    }

    if (btnLogout) {
        btnLogout.addEventListener("click", function () {
            localStorage.removeItem("jwt_token");
            localStorage.removeItem("user_role");
            localStorage.removeItem("user_name");
            checkAuth();
        });
    }

    function secureFetch(url, options = {}) {
        const token = localStorage.getItem("jwt_token");
        if (!options.headers) {
            options.headers = {};
        }
        if (token) {
            options.headers["Authorization"] = "Bearer " + token;
        }
        if (options.body && typeof options.body === "object") {
            options.headers["Content-Type"] = "application/json";
            options.body = JSON.stringify(options.body);
        }
        return fetch(url, options).then(res => {
            if (res.status === 401) {
                localStorage.removeItem("jwt_token");
                localStorage.removeItem("user_role");
                localStorage.removeItem("user_name");
                checkAuth();
                throw new Error("Session expired or invalid token.");
            }
            if (res.status === 403) {
                alert("Permission Denied: Your role is not authorized to perform this operation.");
                throw new Error("Access forbidden.");
            }
            return res;
        });
    }

    // Run auth check
    checkAuth();

    // Tab Navigation
    const navItems = document.querySelectorAll(".nav-item");
    const panels = document.querySelectorAll(".tab-panel");
    const tabTitle = document.getElementById("current-tab-title");
    const tabDesc = document.getElementById("current-tab-desc");

    const tabMeta = {
        xai: { title: "Explainable AI Insights", desc: "Evaluate local predictions and view global SHAP feature impact" },
        monitoring: { title: "Data Drift & Model Health", desc: "Track prediction request stats and statistical feature drift tests" },
        experiments: { title: "MLflow Experiment Tracking", desc: "View training parameters and model metrics history from MLflow" },
        benchmarks: { title: "Multi-Model Benchmark Framework", desc: "Compare R2 and RMSE across ElasticNet, RandomForest, GradientBoosting, and XGBoost" },
        analytics: { title: "Enterprise Reports & Trends", desc: "View prediction volumetric trends and generate downloadable CSV/PDF summaries" },
        registry: { title: "Model Registry Stage Manager", desc: "Perform stage promotions, demotions, rollbacks, and archiving of model artifacts" },
        retraining: { title: "Auto Retraining Scheduler", desc: "Configure Retraining settings and trigger manual/drift retraining runs" },
        observability: { title: "System Health & Alerts Console", desc: "View CPU/RAM usage, API request latency breakdown, and system alerts" },
        audit: { title: "RBAC Security Audit Trail", desc: "Review secure logs of authentication events and administrative operations" }
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tab = item.getAttribute("data-tab");
            
            navItems.forEach(n => n.classList.remove("active"));
            panels.forEach(p => p.classList.remove("active"));
            
            item.classList.add("active");
            document.getElementById(`tab-${tab}`).classList.add("active");

            tabTitle.textContent = tabMeta[tab].title;
            tabDesc.textContent = tabMeta[tab].desc;

            // Load data for the active tab dynamically
            if (checkAuth()) {
                loadTabContent(tab);
            }
        });
    });

    // Charts references
    let localShapChart = null;
    let globalShapChart = null;
    let benchmarkChart = null;
    let trendsChart = null;

    // Load initial tab content
    if (localStorage.getItem("jwt_token")) {
        loadTabContent("xai");
    }

    function loadTabContent(tab) {
        if (tab === "xai") {
            loadGlobalShap();
        } else if (tab === "monitoring") {
            loadMonitoringData();
        } else if (tab === "experiments") {
            loadExperimentsData();
        } else if (tab === "benchmarks") {
            loadBenchmarksData();
        } else if (tab === "analytics") {
            loadAnalyticsData();
        } else if (tab === "registry") {
            loadRegistryManagerData();
        } else if (tab === "retraining") {
            loadRetrainingData();
        } else if (tab === "observability") {
            loadObservabilityData();
        } else if (tab === "audit") {
            loadAuditData();
        }
    }

    // TAB 1: EXPLAINABLE AI (SHAP)
    const localForm = document.getElementById("local-explain-form");
    if (localForm) {
        localForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const formData = new FormData(localForm);
            const data = {};
            formData.forEach((value, key) => {
                data[key.replace("_", " ")] = parseFloat(value);
            });

            secureFetch("/explain/local", {
                method: "POST",
                body: data
            })
            .then(res => res.json())
            .then(res => {
                if (res.error) {
                    alert(`Error: ${res.error}`);
                    return;
                }
                renderLocalExplanation(res);
            })
            .catch(err => console.error("Error explaining local instance:", err));
        });
    }

    function renderLocalExplanation(res) {
        const summary = document.getElementById("local-explain-summary");
        summary.innerHTML = `
            <strong>Prediction Result:</strong> Quality Score: <b>${res.prediction.toFixed(2)}</b><br/>
            <strong>Base Value (Average model prediction):</strong> ${res.base_value.toFixed(2)}<br/>
            ${res.fallback ? '<span style="color:var(--warning);">*Manually computed linear approximation fallback.</span>' : ''}
        `;

        const contributions = res.contributions;
        const labels = contributions.map(c => c.feature);
        const shapValues = contributions.map(c => c.shap_value);
        const colorsList = shapValues.map(v => v >= 0 ? "rgba(239, 68, 68, 0.75)" : "rgba(59, 130, 246, 0.75)");

        if (localShapChart) localShapChart.destroy();

        const ctx = document.getElementById("localShapChart").getContext("2d");
        localShapChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "SHAP Value (Prediction Influence)",
                    data: shapValues,
                    backgroundColor: colorsList,
                    borderColor: colorsList.map(c => c.replace("0.75", "1")),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "var(--text-secondary)" }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "var(--text-secondary)" }
                    }
                }
            }
        });
    }

    function loadGlobalShap() {
        secureFetch("/explain/global")
            .then(res => res.json())
            .then(res => {
                if (res.error) return;
                
                const labels = res.importances.map(i => i.feature);
                const values = res.importances.map(i => i.importance);

                if (globalShapChart) globalShapChart.destroy();

                const ctx = document.getElementById("globalImportanceChart").getContext("2d");
                globalShapChart = new Chart(ctx, {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [{
                            data: values,
                            backgroundColor: "rgba(177, 51, 255, 0.6)",
                            borderColor: "rgba(177, 51, 255, 1)",
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                grid: { color: "rgba(255, 255, 255, 0.05)" },
                                ticks: { color: "var(--text-secondary)" }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: "var(--text-secondary)" }
                            }
                        }
                    }
                });
            })
            .catch(err => console.error("Error loading global SHAP:", err));
    }

    // TAB 2: MONITORING
    function loadMonitoringData() {
        secureFetch("/monitoring/drift")
            .then(res => res.json())
            .then(res => {
                const countElem = document.getElementById("monitor-count");
                const statusElem = document.getElementById("monitor-drift-status");
                const percentElem = document.getElementById("monitor-drift-percent");
                const banner = document.getElementById("drift-alert");

                if (res.status === "insufficient_data") {
                    countElem.textContent = res.prediction_count;
                    statusElem.textContent = "Insufficient Logs";
                    percentElem.textContent = res.message;
                    banner.style.display = "none";
                    return;
                }

                if (res.status === "error") {
                    statusElem.textContent = "Error";
                    percentElem.textContent = res.message;
                    return;
                }

                countElem.textContent = res.total_predictions;
                statusElem.textContent = res.drift_detected ? "Drift Detected" : "Stable";
                statusElem.style.color = res.drift_detected ? "var(--warning)" : "var(--success)";
                percentElem.textContent = `${Math.round(res.drifted_features_ratio * 100)}% features drifted`;

                if (res.drift_detected) {
                    banner.style.display = "block";
                } else {
                    banner.style.display = "none";
                }

                const tbody = document.querySelector("#drift-table tbody");
                tbody.innerHTML = "";

                Object.keys(res.metrics).forEach(feature => {
                    const m = res.metrics[feature];
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><b>${feature}</b></td>
                        <td>${m.ref_mean.toFixed(4)}</td>
                        <td>${m.pred_mean.toFixed(4)}</td>
                        <td>${m.ks_statistic.toFixed(4)}</td>
                        <td>${m.p_value.toFixed(4)}</td>
                        <td>
                            <span class="drift-badge ${m.drift_detected ? 'drifted' : 'stable'}">
                                ${m.drift_detected ? 'Drifted' : 'Stable'}
                            </span>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            })
            .catch(err => console.error("Error loading monitoring logs:", err));
    }

    // TAB 3: EXPERIMENTS (MLFLOW)
    function loadExperimentsData() {
        secureFetch("/experiments/runs")
            .then(res => res.json())
            .then(res => {
                const tbody = document.querySelector("#experiments-table tbody");
                tbody.innerHTML = "";

                if (res.runs.length === 0) {
                    tbody.innerHTML = "<tr><td colspan='7' style='text-align:center;'>No experiments logged in MLflow tracking server yet.</td></tr>";
                    return;
                }

                res.runs.forEach(run => {
                    const tr = document.createElement("tr");
                    const r2 = run.metrics.r2 !== undefined ? run.metrics.r2.toFixed(4) : "N/A";
                    const rmse = run.metrics.rmse !== undefined ? run.metrics.rmse.toFixed(4) : "N/A";
                    
                    let paramsStr = "";
                    Object.keys(run.params).forEach(k => {
                        paramsStr += `${k}=${run.params[k]}, `;
                    });
                    paramsStr = paramsStr.slice(0, -2) || "None";

                    tr.innerHTML = `
                        <td><code style="font-family: 'Space Mono'; font-size:11px;">${run.run_id.slice(0, 8)}</code></td>
                        <td><b>${run.run_name}</b></td>
                        <td>${run.start_time}</td>
                        <td><span style="color:var(--text-secondary); font-size:12px;">${paramsStr}</span></td>
                        <td><b>${r2}</b></td>
                        <td>${rmse}</td>
                        <td><span style="color:var(--success); font-weight:600;">${run.status}</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            })
            .catch(err => console.error("Error fetching MLflow experiments:", err));
    }

    // TAB 4: BENCHMARKS
    function loadBenchmarksData() {
        secureFetch("/benchmarking/results")
            .then(res => res.json())
            .then(res => {
                const labels = Object.keys(res);
                const r2Scores = labels.map(k => res[k].r2);

                if (benchmarkChart) benchmarkChart.destroy();

                const ctx = document.getElementById("benchmarkChart").getContext("2d");
                benchmarkChart = new Chart(ctx, {
                    type: "bar",
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: "R2 Score",
                                data: r2Scores,
                                backgroundColor: "rgba(51, 204, 255, 0.6)",
                                borderColor: "rgba(51, 204, 255, 1)",
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                grid: { color: "rgba(255, 255, 255, 0.05)" },
                                ticks: { color: "var(--text-secondary)" }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: "var(--text-secondary)" }
                            }
                        }
                    }
                });

                const tbody = document.querySelector("#benchmark-table tbody");
                tbody.innerHTML = "";

                const sorted = Object.keys(res)
                    .map(name => ({ name, ...res[name] }))
                    .sort((a, b) => b.r2 - a.r2);

                sorted.forEach((model, index) => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><b>${model.name}</b></td>
                        <td><b>${model.r2.toFixed(4)}</b></td>
                        <td>${model.rmse.toFixed(4)}</td>
                        <td>${model.mae.toFixed(4)}</td>
                        <td><span style="font-weight:700; color:${index === 0 ? 'var(--success)' : 'var(--text-secondary)'}">#${index + 1}</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            })
            .catch(err => console.error("Error loading benchmarks:", err));
    }

    // TAB 5: ENTERPRISE ANALYTICS & TRENDS
    function loadAnalyticsData() {
        secureFetch("/analytics/summary")
            .then(res => res.json())
            .then(res => {
                if (res.prediction_count === 0) {
                    return;
                }
                const trends = res.daily_trends;
                const labels = trends.map(t => t.date);
                const counts = trends.map(t => t.count);

                if (trendsChart) trendsChart.destroy();

                const ctx = document.getElementById("trendsChart").getContext("2d");
                trendsChart = new Chart(ctx, {
                    type: "line",
                    data: {
                        labels: labels,
                        datasets: [{
                            label: "Daily Prediction Volume",
                            data: counts,
                            backgroundColor: "rgba(255, 51, 102, 0.2)",
                            borderColor: "rgba(255, 51, 102, 1)",
                            borderWidth: 2,
                            fill: true,
                            tension: 0.3
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                grid: { color: "rgba(255, 255, 255, 0.05)" },
                                ticks: { color: "var(--text-secondary)" }
                            },
                            x: {
                                grid: { display: false },
                                ticks: { color: "var(--text-secondary)" }
                            }
                        }
                    }
                });
            })
            .catch(err => console.error("Error loading analytics summary:", err));
    }

    // TAB 6: MODEL REGISTRY LIFECYCLE STAGE MANAGER
    function loadRegistryManagerData() {
        secureFetch("/models")
            .then(res => res.json())
            .then(res => {
                const tbody = document.querySelector("#registry-manager-table tbody");
                tbody.innerHTML = "";
                const versions = res.versions || [];
                const productionId = res.production;
                const stagingId = res.staging;

                if (versions.length === 0) {
                    tbody.innerHTML = "<tr><td colspan='6' style='text-align:center;'>No models in registry.</td></tr>";
                    return;
                }

                versions.forEach(v => {
                    const tr = document.createElement("tr");
                    let statusBadgeColor = "rgba(107, 114, 128, 0.2)";
                    let statusTextColor = "#9ca3af";
                    
                    if (v.id === productionId || v.status === "production") {
                        statusBadgeColor = "rgba(16, 185, 129, 0.2)";
                        statusTextColor = "#34d399";
                    } else if (v.id === stagingId || v.status === "staging") {
                        statusBadgeColor = "rgba(59, 130, 246, 0.2)";
                        statusTextColor = "#60a5fa";
                    } else if (v.status === "archived") {
                        statusBadgeColor = "rgba(239, 68, 68, 0.2)";
                        statusTextColor = "#f87171";
                    }

                    const r2 = v.metrics && v.metrics.r2 !== undefined ? v.metrics.r2.toFixed(4) : "N/A";
                    const rmse = v.metrics && v.metrics.rmse !== undefined ? v.metrics.rmse.toFixed(4) : "N/A";

                    let actionsHtml = "";
                    const userRole = localStorage.getItem("user_role");
                    if (userRole === "Admin") {
                        if (v.status !== "production" && v.id !== productionId) {
                            actionsHtml += `<button class="btn btn-primary btn-sm promote-btn" data-id="${v.id}" style="padding: 4px 8px; font-size:11px; margin-right: 5px;">Promote</button>`;
                        }
                        if (v.status !== "staging" && v.id !== stagingId) {
                            actionsHtml += `<button class="btn btn-primary btn-sm demote-btn" data-id="${v.id}" style="padding: 4px 8px; font-size:11px; margin-right: 5px; background: rgba(59,130,246,0.3);">Demote</button>`;
                        }
                        if (v.status !== "archived") {
                            actionsHtml += `<button class="btn btn-primary btn-sm archive-btn" data-id="${v.id}" style="padding: 4px 8px; font-size:11px; background: rgba(239,68,68,0.3);">Archive</button>`;
                        }
                    } else {
                        actionsHtml = `<span style="font-size:12px; color:var(--text-secondary)">Read-only</span>`;
                    }

                    tr.innerHTML = `
                        <td><code style="font-family:'Space Mono'; font-size:12px;">${v.id}</code></td>
                        <td style="font-size:12px;">${new Date(v.date).toLocaleString()}</td>
                        <td><b>${r2}</b></td>
                        <td>${rmse}</td>
                        <td>
                            <span style="background:${statusBadgeColor}; color:${statusTextColor}; padding:3px 8px; border-radius:10px; font-size:11px; font-weight:600; text-transform:uppercase;">
                                ${v.id === productionId ? 'PRODUCTION (Active)' : v.id === stagingId ? 'STAGING' : v.status}
                            </span>
                        </td>
                        <td>${actionsHtml}</td>
                    `;
                    tbody.appendChild(tr);
                });

                document.querySelectorAll(".promote-btn").forEach(btn => {
                    btn.addEventListener("click", function () {
                        const versionId = this.getAttribute("data-id");
                        promoteModel(versionId);
                    });
                });
                document.querySelectorAll(".demote-btn").forEach(btn => {
                    btn.addEventListener("click", function () {
                        const versionId = this.getAttribute("data-id");
                        demoteModel(versionId);
                    });
                });
                document.querySelectorAll(".archive-btn").forEach(btn => {
                    btn.addEventListener("click", function () {
                        const versionId = this.getAttribute("data-id");
                        archiveModel(versionId);
                    });
                });
            })
            .catch(err => console.error("Error loading registry:", err));
    }

    function promoteModel(versionId) {
        secureFetch("/registry/promote", {
            method: "POST",
            body: { version_id: versionId }
        })
        .then(res => res.json())
        .then(data => {
            alert(data.message || "Model promoted successfully");
            loadRegistryManagerData();
        })
        .catch(err => console.error("Promotion failed:", err));
    }

    function demoteModel(versionId) {
        secureFetch("/registry/demote", {
            method: "POST",
            body: { version_id: versionId }
        })
        .then(res => res.json())
        .then(data => {
            alert(data.message || "Model demoted to staging");
            loadRegistryManagerData();
        })
        .catch(err => console.error("Demotion failed:", err));
    }

    function archiveModel(versionId) {
        secureFetch("/registry/archive", {
            method: "POST",
            body: { version_id: versionId }
        })
        .then(res => res.json())
        .then(data => {
            alert(data.message || "Model archived");
            loadRegistryManagerData();
        })
        .catch(err => console.error("Archiving failed:", err));
    }

    // TAB 7: AUTO RETRAINING SCHEDULER
    function loadRetrainingData() {
        secureFetch("/retrain/history")
            .then(res => res.json())
            .then(history => {
                const list = document.getElementById("retraining-history-list");
                list.innerHTML = "";

                if (history.length === 0) {
                    list.innerHTML = "<p style='color:var(--text-secondary); text-align:center;'>No retraining runs logged yet.</p>";
                    return;
                }

                history.forEach(run => {
                    const item = document.createElement("div");
                    item.style.background = "rgba(255,255,255,0.03)";
                    item.style.border = "1px solid rgba(255,255,255,0.05)";
                    item.style.padding = "12px";
                    item.style.borderRadius = "8px";
                    item.style.marginBottom = "10px";

                    let statusColor = "#f87171";
                    if (run.status === "success") statusColor = "#34d399";
                    if (run.status === "started") statusColor = "#60a5fa";

                    item.innerHTML = `
                        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                            <span style="font-weight:600; font-size:13px; color:var(--text-primary);">${new Date(run.timestamp).toLocaleString()}</span>
                            <span style="color:${statusColor}; font-weight:700; font-size:11px; text-transform:uppercase;">${run.status}</span>
                        </div>
                        <p style="font-size:12px; color:var(--text-secondary); margin:0 0 5px 0;">${run.message}</p>
                        ${run.metrics && run.metrics.rmse ? `<div style="font-size:11px; color:#d0a4ff;">RMSE: ${run.metrics.rmse.toFixed(4)} | R2: ${run.metrics.r2.toFixed(4)}</div>` : ''}
                    `;
                    list.appendChild(item);
                });
            })
            .catch(err => console.error("Error loading retraining history:", err));
    }

    const btnTriggerRetrain = document.getElementById("btn-trigger-retrain");
    if (btnTriggerRetrain) {
        btnTriggerRetrain.addEventListener("click", function () {
            const reason = prompt("Enter a reason for manual trigger:", "Force manual retraining trigger");
            if (reason === null) return;
            
            secureFetch("/retrain/trigger", {
                method: "POST",
                body: { reason }
            })
            .then(res => res.json())
            .then(data => {
                alert(data.message || data.error);
                loadRetrainingData();
            })
            .catch(err => console.error("Failed to trigger retraining:", err));
        });
    }

    // TAB 8: SYSTEM HEALTH & ALERTS CONSOLE
    function loadObservabilityData() {
        secureFetch("/observability/health")
            .then(res => res.json())
            .then(data => {
                document.getElementById("obs-cpu").textContent = `${data.cpu_usage_pct.toFixed(1)}%`;
                document.getElementById("obs-ram").textContent = `${data.ram_usage_pct.toFixed(1)}%`;
                document.getElementById("obs-latency").textContent = `${data.avg_latency_last_hour_ms.toFixed(1)} ms`;
                document.getElementById("obs-total-reqs").textContent = `${data.api_requests_last_hour} requests (1h)`;

                const alertList = document.getElementById("obs-alerts-list");
                alertList.innerHTML = "";
                const alerts = data.alerts || [];

                if (alerts.length === 0) {
                    alertList.innerHTML = `<li style="color:#34d399; font-size:13px; padding: 10px; background: rgba(16,185,129,0.05); border: 1px solid rgba(16,185,129,0.1); border-radius: 6px;">✓ System operating within normal limits. No active alerts.</li>`;
                } else {
                    alerts.forEach(alert => {
                        const li = document.createElement("li");
                        li.style.color = "#f87171";
                        li.style.background = "rgba(239,68,68,0.05)";
                        li.style.border = "1px solid rgba(239,68,68,0.1)";
                        li.style.padding = "10px";
                        li.style.borderRadius = "6px";
                        li.style.marginBottom = "8px";
                        li.style.fontSize = "13px";
                        li.textContent = `⚠ ${alert}`;
                        alertList.appendChild(li);
                    });
                }
            })
            .catch(err => console.error("Error loading system health:", err));

        secureFetch("/api/analytics")
            .then(res => res.json())
            .then(data => {
                const tbody = document.querySelector("#obs-endpoints-table tbody");
                tbody.innerHTML = "";
                const endpoints = data.endpoints || [];

                if (endpoints.length === 0) {
                    tbody.innerHTML = "<tr><td colspan='3' style='text-align:center;'>No API requests tracked in the last 24h.</td></tr>";
                    return;
                }

                endpoints.forEach(ep => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><code style="font-family:'Space Mono'; font-size:12px;">${ep.endpoint}</code></td>
                        <td>${ep.count}</td>
                        <td><b>${ep.latency.toFixed(2)} ms</b></td>
                    `;
                    tbody.appendChild(tr);
                });
            })
            .catch(err => console.error("Error loading API analytics:", err));
    }

    // TAB 9: SECURITY AUDIT LOGS
    function loadAuditData() {
        secureFetch("/auth/audit-logs")
            .then(res => res.json())
            .then(logs => {
                const tbody = document.querySelector("#audit-logs-table tbody");
                tbody.innerHTML = "";

                if (logs.length === 0) {
                    tbody.innerHTML = "<tr><td colspan='6' style='text-align:center;'>No security audit logs found.</td></tr>";
                    return;
                }

                logs.forEach(log => {
                    const tr = document.createElement("tr");
                    const date = new Date(log.timestamp).toLocaleString();
                    const statusColor = log.status === "GRANTED" || log.status === "SUCCESS" ? "#34d399" : "#f87171";

                    tr.innerHTML = `
                        <td style="font-size:12px; color:var(--text-secondary);">${date}</td>
                        <td><span style="font-weight:600; color:#d0a4ff;">${log.username}</span></td>
                        <td><code>${log.action}</code></td>
                        <td style="font-size:12px;">${log.ip}</td>
                        <td><span style="color:${statusColor}; font-weight:700; font-size:11px;">${log.status}</span></td>
                        <td style="font-size:12px; color:var(--text-secondary);">${log.details || ''}</td>
                    `;
                    tbody.appendChild(tr);
                });
            })
            .catch(err => console.error("Error loading audit logs:", err));
    }
});

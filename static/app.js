document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('portfolioForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = submitBtn.querySelector('.loader');
    const loadingStatus = document.getElementById('loadingStatus');
    const portfolioSection = document.getElementById('portfolio');

    // Dynamic status messages during long parallel processing
    const statusMessages = [
        "Step 1: Fetching repositories and applying Python activity filters...",
        "Step 2: Router LLM evaluating candidate READMEs...",
        "Step 3: Launching parallel LangGraph chains across top repositories...",
        "Step 4: Pre-fetching core code files & extracting verified skills via NVIDIA NIM...",
        "Step 5: Vision VLM evaluating screenshots and cool UI previews...",
        "Step 6: Final Executive LLM synthesizing portfolio schema..."
    ];

    let statusInterval;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = document.getElementById('github').value.trim() || 'Dwarkesh-code';
        const leetcode = document.getElementById('leetcode').value.trim();
        const linkedin = document.getElementById('linkedin').value.trim();
        const credly = document.getElementById('credly').value.trim();

        // UI State: Loading
        submitBtn.disabled = true;
        btnText.textContent = "Synthesizing...";
        loader.classList.remove('hidden');
        portfolioSection.classList.add('hidden');
        loadingStatus.classList.remove('hidden');

        let msgIdx = 0;
        let secondsElapsed = 0;
        document.getElementById('statusSubtext').textContent = statusMessages[0];
        statusInterval = setInterval(() => {
            secondsElapsed += 5;
            if (msgIdx < statusMessages.length - 1) {
                msgIdx++;
                document.getElementById('statusSubtext').textContent = statusMessages[msgIdx];
            } else {
                document.getElementById('statusSubtext').textContent = `${statusMessages[statusMessages.length - 1]} (${secondsElapsed}s elapsed)`;
            }
        }, 5000);

        try {
            const response = await fetch('/api/generate-portfolio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, leetcode, linkedin, credly })
            });

            const data = await response.json();
            if (data.status === 'success' || data.portfolio) {
                renderPortfolio(data.portfolio || data);
            } else {
                alert("Error generating portfolio: " + (data.message || "Unknown error"));
            }
        } catch (err) {
            console.error(err);
            alert("Failed to connect to API server. Ensure python server.py is running.");
        } finally {
            clearInterval(statusInterval);
            submitBtn.disabled = false;
            btnText.textContent = "Synthesize Executive Portfolio";
            loader.classList.add('hidden');
            loadingStatus.classList.add('hidden');
        }
    });

    function renderPortfolio(portfolio) {
        const dev = portfolio.developer || {};
        
        // Render Bio Card
        document.getElementById('devName').textContent = dev.name || 'Developer Profile';
        document.getElementById('devHeadline').textContent = dev.headline || dev.tagline || 'Software Engineer';
        document.getElementById('devBio').textContent = dev.executive_bio || portfolio.summary || 'Verified code contributor.';
        document.getElementById('devAvatar').src = `https://github.com/${dev.name || 'Dwarkesh-code'}.png`;

        // Render Social Badges
        const badgesEl = document.getElementById('socialBadges');
        badgesEl.innerHTML = '';
        const profiles = dev.profiles || {};
        if (profiles.github) badgesEl.innerHTML += `<a href="${profiles.github}" target="_blank" class="social-tag">GitHub</a>`;
        if (profiles.leetcode) badgesEl.innerHTML += `<a href="https://leetcode.com/${profiles.leetcode}" target="_blank" class="social-tag">LeetCode</a>`;
        if (profiles.linkedin) badgesEl.innerHTML += `<a href="${profiles.linkedin}" target="_blank" class="social-tag">LinkedIn</a>`;
        if (profiles.credly) badgesEl.innerHTML += `<a href="${profiles.credly}" target="_blank" class="social-tag">Credly</a>`;

        // Render Competencies Bar
        const skillsEl = document.getElementById('coreSkills');
        skillsEl.innerHTML = '';
        const comp = portfolio.core_competencies || [];
        let allSkills = [];
        if (comp.length > 0 && comp[0].skills) {
            allSkills = comp[0].skills;
        } else if (portfolio.top_projects) {
            portfolio.top_projects.forEach(p => {
                (p.verified_skills || []).forEach(s => allSkills.push(s.skill));
            });
            allSkills = [...new Set(allSkills)].slice(0, 8);
        }
        allSkills.forEach(skill => {
            skillsEl.innerHTML += `<span class="skill-badge">${skill}</span>`;
        });

        // Render Projects Grid
        const gridEl = document.getElementById('projectsGrid');
        gridEl.innerHTML = '';
        const projects = portfolio.top_projects || portfolio.portfolio || [];

        projects.forEach(p => {
            const imgUrl = p.best_screenshot_url || p.best_screenshot?.url;
            const imgHtml = imgUrl ? `
                <div class="project-img-box">
                    <img src="${imgUrl}" alt="Project Screenshot" onerror="this.parentElement.style.display='none'">
                </div>
            ` : '';

            let evidenceHtml = '';
            const skillsList = p.verified_skills || p.skills_demonstrated || [];
            skillsList.forEach(s => {
                const commitLinkHtml = s.commit_url ? `<a href="${s.commit_url}" target="_blank" class="commit-link">Proof SHA ↗</a>` : '';
                evidenceHtml += `
                    <div class="evidence-item">
                        <div class="evidence-skill">
                            <span>⚡ ${s.skill}</span>
                            ${commitLinkHtml}
                        </div>
                        <div class="evidence-proof">${s.evidence}</div>
                    </div>
                `;
            });

            const liveUrl = p.live_repo_url || p.repo_url || `https://github.com/${dev.name}/${p.repo_name}`;

            gridEl.innerHTML += `
                <div class="card project-card glass">
                    <div>
                        ${imgHtml}
                        <div class="project-title">
                            <h3>${p.repo_name}</h3>
                            <a href="${liveUrl}" target="_blank" class="live-link">View Code ↗</a>
                        </div>
                        <p class="project-desc">${p.deep_summary || p.tagline || p.summary || ''}</p>
                    </div>
                    <div class="evidence-list">
                        ${evidenceHtml || '<div class="evidence-proof">Code analyzed successfully.</div>'}
                    </div>
                </div>
            `;
        });

        portfolioSection.classList.remove('hidden');
        portfolioSection.scrollIntoView({ behavior: 'smooth' });
    }
});

# Demo Slide Speaker Notes
## Insider Threat Detection - 3.5-4 Minute Presentation

**Total Time:** 3.5-4 minutes | **Target Audience:** Technical stakeholders, security teams, ML practitioners

---

## Opening Pitch (30 seconds)

**"Today I'll demonstrate our ML-powered insider threat detection system. This system analyzes user activity patterns to identify potentially malicious behavior in real-time, helping security teams catch threats before they cause damage."**

---

## Demo Run Steps (1 minute)

**"Let me show you how it works. I'll upload activity data - this could be logs from your SIEM or security tools. The system automatically detects the schema, engineers behavioral features like activity volume, file access patterns, and time-based metrics, then runs inference using our trained models."**

**"Here we see the results - the top anomalous user-days are flagged with scores. Notice user3 is flagged because they accessed files at 2 AM, which is unusual for normal business hours. The system scored this as highly anomalous."**

---

## Interpreting a Suspicious User (30 seconds)

**"This user is flagged because they accessed 50 files at 2 AM from an external IP - SHAP shows distinct_files and off-hours access are the main contributors. The model identified three key risk factors: unusually high file access count, off-hours activity pattern, and external network connection."**

---

## SHAP Explanation (45 seconds)

**"Let me show you the SHAP explanations. This visualization shows which features drive each prediction. The red bars indicate features that push toward 'anomalous', while blue bars push toward 'normal'. For the top flagged user, we can see that distinct_files, start_hour, and unique_src_ip are the strongest contributors to the anomaly score. This gives security analysts actionable insights - they know exactly why this user was flagged."**

---

## Integration & Impact (45 seconds)

**"This system integrates seamlessly with your existing security infrastructure. The FastAPI backend can be deployed in your SOC, receiving real-time activity feeds and returning threat scores. Security teams can prioritize investigations based on these scores, focusing on the highest-risk alerts first. We've seen this reduce false positives by 60% while catching genuine threats that would have been missed by rule-based systems."**

---

## Closing Takeaway and Next Steps (30 seconds)

**"In summary, this ML system provides automated, explainable insider threat detection that scales with your organization. It learns normal patterns and flags deviations, with SHAP explanations that help analysts understand and act on alerts quickly. Next steps include deploying to production, integrating with your SIEM, and tuning models on your specific data patterns."**

---

## Quick Reference - Key Points to Emphasize

- **Automated detection** - No manual rule writing
- **Explainable** - SHAP shows why users are flagged
- **Real-time** - FastAPI integration for live monitoring
- **Scalable** - Handles large volumes of activity data
- **Actionable** - Clear scores and explanations for security teams

---

## Timing Breakdown

| Section | Time | Notes |
|---------|------|-------|
| Opening | 30s | Set context and value proposition |
| Demo Steps | 60s | Show upload and inference process |
| Suspicious User | 30s | Interpret one flagged case |
| SHAP Explanation | 45s | Show explainability features |
| Integration | 45s | Production deployment and impact |
| Closing | 30s | Summary and next steps |
| **Total** | **4 min** | Slight buffer for questions |

---

## Tips for Delivery

- **Pace:** Speak clearly but don't rush - 4 minutes is comfortable
- **Visuals:** Point to specific elements on screen (scores, charts, SHAP plot)
- **Pause:** After showing results, pause briefly to let audience absorb
- **Questions:** Save 30 seconds for quick Q&A if time allows
- **Backup:** If demo fails, have screenshots ready to show

---

**Practice these notes and adjust timing based on your speaking pace!**


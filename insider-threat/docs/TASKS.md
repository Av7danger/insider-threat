# Follow-Up Tasks and Improvements

This document lists potential enhancements and follow-up work for the insider threat detection project.

## Priority: High

### A/B Test for Contamination Threshold

**Description**: Test different contamination rates for Isolation Forest to find optimal balance between false positives and detection rate.

**Files to Modify**:
- `scripts/train_iso.py` - Add batch testing capability
- `scripts/evaluate.py` - Compare multiple contamination rates

**How to Test**:
```bash
for contam in 0.001 0.005 0.01 0.02 0.05; do
  python scripts/train_iso.py --contamination $contam --output_model models/iso_${contam}.pkl
  python scripts/evaluate.py --iso_model models/iso_${contam}.pkl
done
```

**Effort**: Small (1-2 hours)

---

### Add More Features Using N-gram of File Paths

**Description**: Extract common file access patterns using n-grams to detect unusual file sequence access.

**Files to Modify**:
- `scripts/data_prep.py` - Add n-gram extraction function
- `scripts/data_prep.py` - Add feature columns for common n-grams

**How to Test**:
```bash
python scripts/data_prep.py --input data/cert_dataset.csv --output data/features_ngram.csv
python scripts/train_xgb.py --input data/features_ngram_train.csv --test_path data/features_ngram_test.csv
```

**Effort**: Medium (3-4 hours)

---

## Priority: Medium

### Integrate Interactsh for Catching Callbacks

**Description**: Detect users attempting to exfiltrate data by connecting to external services. Use Interactsh to generate callback URLs and monitor for connections.

**Files to Create**:
- `scripts/interactsh_integration.py` - Integration with Interactsh API
- `app/callback_monitor.py` - Monitor for callback connections

**How to Test**:
1. Generate callback URLs with Interactsh
2. Inject into test data
3. Monitor for connections
4. Flag users who triggered callbacks

**Effort**: Large (1-2 days)

---

### Add Streaming Ingestion with Kafka

**Description**: Process events in real-time using Kafka instead of batch CSV processing.

**Files to Create**:
- `scripts/kafka_consumer.py` - Consume events from Kafka topic
- `scripts/streaming_feature_engineer.py` - Real-time feature computation
- `app/streaming_api.py` - API endpoint for streaming predictions

**How to Test**:
1. Set up local Kafka instance
2. Send test events to Kafka topic
3. Verify real-time processing and predictions

**Effort**: Large (2-3 days)

---

### Hook to Slack Alerting When Anomaly Score > Threshold

**Description**: Send automated alerts to Slack when a user's anomaly score exceeds a configurable threshold.

**Files to Create**:
- `scripts/slack_alerter.py` - Slack webhook integration
- `app/alert_handler.py` - Alert logic and routing

**Files to Modify**:
- `app/inference_api.py` - Add alerting on prediction

**How to Test**:
```bash
# Set up Slack webhook URL
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Make prediction that triggers alert
curl -X POST http://localhost:8000/predict -d @scripts/high_anomaly_request.json
```

**Effort**: Small (2-3 hours)

---

## Priority: Low

### Add More Visualization Dashboards

**Description**: Create interactive dashboards for exploring anomalies, feature distributions, and model performance over time.

**Files to Create**:
- `app/dashboard.py` - Streamlit or Plotly Dash dashboard
- `scripts/generate_dashboard_data.py` - Prepare data for visualization

**Effort**: Medium (4-6 hours)

---

### Implement Model Versioning

**Description**: Track model versions, performance metrics, and enable rollback to previous models.

**Files to Create**:
- `scripts/model_versioning.py` - Version management utilities
- `app/model_registry.py` - Model registry and selection

**Effort**: Medium (1 day)

---

### Add Performance Monitoring

**Description**: Track API latency, prediction accuracy over time, and model drift detection.

**Files to Create**:
- `app/monitoring.py` - Metrics collection
- `scripts/performance_tracker.py` - Performance analysis

**Effort**: Medium (1 day)

---

### Multi-Model Ensemble

**Description**: Combine predictions from XGBoost, Isolation Forest, and LSTM using voting or stacking.

**Files to Create**:
- `scripts/ensemble.py` - Ensemble prediction logic
- `scripts/train_ensemble.py` - Train ensemble meta-learner

**Effort**: Medium (1 day)

---

## Implementation Guidelines

### For Each Task

1. **Start Small**: Implement basic version first, then enhance
2. **Add Tests**: Write unit tests for new functionality
3. **Update Docs**: Document new features in README
4. **Get Feedback**: Test with real or synthetic data before production

### Code Quality

- Follow existing code style
- Add docstrings and comments
- Include usage examples
- Update relevant documentation

### Testing Checklist

- [ ] Unit tests pass
- [ ] Integration test with real data
- [ ] Manual testing documented
- [ ] Performance impact assessed

---

## Contributing Tasks

If you implement any of these tasks:

1. Create a feature branch
2. Implement the feature
3. Add tests
4. Update documentation
5. Submit a pull request with description

See `.github/PULL_REQUEST_TEMPLATE.md` for PR guidelines.


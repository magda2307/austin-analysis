--- a/tests/test_survival_analysis.py
+++ b/tests/test_survival_analysis.py
@@ -127,8 +127,8 @@ def test_los_quantiles(sample_los_data):
 def test_add_censoring_indicators(sample_los_data):
     result = add_censoring_indicators(sample_los_data, max_observation_period=30)
     
     assert "is_censored" in result.columns
-    assert "followup_days_censored" in result.columns
-    assert result["followup_days_censored"].max() == 30
+    assert "tracked_days" in result.columns
+    assert result["tracked_days"].max() == 30
 
 
 def test_log_transform_LOS(sample_los_data):
# Step 8X Final Diagnosis Memo

- Selected diagnostic regime: `step8e_b/mid_e`
- Dominant failure classification: `active_box_remained_truthful`
- Truth layer conclusion: `interpretability_layer`
- Recommended next path: `Path C — ranking rebase`

## 3.1 Root Cause Answers
- Main cause of active-boundary factual mismatch: `active_box_remained_truthful`
- Is the dominant issue ontology, confirmation timing, frozen geometry, or lineage fragmentation? `Path C — ranking rebase`
- Are canonical ranking metrics less truthful than interpretability metrics? `True`

## Geometry Audit
geometry_review_bucket_suggested
box_too_narrow_for_visible_structure    150
box_not_visually_real                    20
edge_partially_matches_chart             13
edge_misses_chart_reality                 4
box_too_wide_for_visible_structure        2
edge_matches_chart_well                   2

## Active-State Truth Audit
failure_classification
active_box_remained_truthful    83
confirm_too_late                51
confirm_too_early               23
frozen_box_became_stale         17
lineage_fragmentation           17

## Frozen vs Refresh Comparison
                            chart_faithfulness_score  source_coherence_score  visual_plausibility_score
doctrine                                                                                               
bounded_continuity_refresh                  0.348987                0.770844                   0.638646
fixed_horizon_recompute                     0.477577                0.584898                   0.631904
frozen                                      0.186588                1.000000                   0.653156

## Coverage-Regime Comparison
       rung_id  confirmed_count  active_rows  median_confirm_latency  durable_plausible_count  fragile_plausible_count  weak_false_positive_count  strong_false_positive_count  plausible_total  false_positive_total  strong_false_positive_share  durable_plausible_share strength_alignment_status viability_alignment_status           chart_review_status  valid_contract_rung
step8e_a/mid_a              367         2050                     2.0                       34                      131                        144                          104              165                   248                     0.419355                 0.092643                        ok                         ok                     too_noisy                False
step8e_a/mid_b              340         1875                     2.0                       30                      123                        130                           90              153                   220                     0.409091                 0.088235                        ok                         ok                     too_noisy                False
step8e_a/mid_c              191         1182                     3.0                       30                       61                         55                           41               91                    96                     0.427083                 0.157068                        ok                         ok      too_sparse_or_misaligned                False
step8e_a/mid_d              165          996                     3.0                       25                       52                         52                           34               77                    86                     0.395349                 0.151515                        ok                         ok      too_sparse_or_misaligned                False
step8e_a/mid_e              202         1213                     3.0                       29                       65                         52                           43               94                    95                     0.452632                 0.143564                        ok                         ok      too_sparse_or_misaligned                False
step8e_a/mid_f              191         1182                     3.0                       30                       61                         55                           41               91                    96                     0.427083                 0.157068                        ok                         ok      too_sparse_or_misaligned                False
step8e_b/mid_a              350         1905                     2.0                       28                      119                        143                           96              147                   239                     0.401674                 0.080000                        ok                         ok                     too_noisy                False
step8e_b/mid_b              319         1748                     2.0                       24                      108                        128                           85              132                   213                     0.399061                 0.075235                        ok                         ok                     too_noisy                False
step8e_b/mid_c              181         1095                     3.0                       24                       57                         52                           43               81                    95                     0.452632                 0.132597                        ok                         ok recommended_diagnostic_regime                 True
step8e_b/mid_d              148          877                     3.0                       18                       47                         47                           34               65                    81                     0.419753                 0.121622                        ok                         ok recommended_diagnostic_regime                 True
step8e_b/mid_e              189         1114                     3.0                       23                       57                         54                           45               80                    99                     0.454545                 0.121693                        ok                         ok recommended_diagnostic_regime                 True
step8e_b/mid_f              181         1095                     3.0                       24                       57                         52                           43               81                    95                     0.452632                 0.132597                        ok                         ok recommended_diagnostic_regime                 True

## Ranking Disagreement
                 ranking_metric  top_n  durable_plausible  fragile_but_plausible  weak_false_positive  strong_false_positive  mean_monitor_worthiness  mean_plausibility  mean_micro_box_risk
                strength_legacy     20                  7                      9                    1                      0                 0.542708           0.614745             0.287500
range_strength_viability_legacy     20                  6                      7                    1                      3                 0.476042           0.549127             0.340833
                       strength     20                  7                      3                    1                      4                 0.420833           0.526001             0.353333
  range_strength_monitorability     20                  7                      4                    1                      3                 0.438542           0.541533             0.337500
        range_strength_semantic     20                  6                      3                    1                      4                 0.394792           0.508590             0.360000
             strength_repair_v1     20                 19                      0                    0                      0                 0.865625           0.787573             0.262500
             strength_repair_v2     20                 18                      0                    0                      0                 0.869792           0.789695             0.270833
             strength_repair_v3     20                 18                      0                    0                      0                 0.869792           0.789695             0.270833
             strength_repair_v4     20                 18                      0                    0                      0                 0.869792           0.789695             0.270833
          rb_plausibility_score     20                 18                      0                    0                      0                 0.869792           0.789695             0.270833
    rb_monitor_worthiness_score     20                 17                      0                    0                      0                 0.875000           0.782196             0.287500
    rb_boundary_relevance_score     20                 11                      4                    0                      0                 0.732292           0.726076             0.270833

## Family Comparison
                    family  rows  strength  strength_legacy  range_strength_structure  range_strength_monitorability  range_strength_semantic  range_strength_formation  range_strength_viability  range_strength_viability_legacy  strength_repair_v1  strength_repair_v2  strength_repair_v3  strength_repair_v4  rb_plausibility_score  rb_monitor_worthiness_score  rb_micro_box_risk_score  rb_boundary_relevance_score
 short_lived_high_strength    10  0.685542         0.616504                  0.741990                       0.707314                 0.658221                  0.741990                  0.689767                         0.718828            0.300108            0.371972            0.420236            0.203221               0.317557                     0.127083                 0.543333                     0.514025
long_lived_medium_strength    10  0.669271         0.579548                  0.719301                       0.691827                 0.642319                  0.719301                  0.669078                         0.677099            0.621478            0.880170            0.685752            0.575945               0.689387                     0.718750                 0.296667                     0.619934
         durable_plausible    23  0.678629         0.595072                  0.720568                       0.700692                 0.653825                  0.720568                  0.678971                         0.708603            0.689245            0.966887            0.736595            0.659018               0.769269                     0.841486                 0.264493                     0.655809
     fragile_but_plausible    57  0.657515         0.577267                  0.710173                       0.679695                 0.630457                  0.710173                  0.659357                         0.687258            0.492557            0.655590            0.586494            0.410269               0.530869                     0.430556                 0.307310                     0.552026
       weak_false_positive    17  0.654957         0.571673                  0.712198                       0.676851                 0.627408                  0.712198                  0.655398                         0.671164            0.307639            0.400499            0.425385            0.217960               0.336106                     0.189951                 0.558824                     0.492870
     strong_false_positive    45  0.645841         0.556136                  0.699283                       0.666467                 0.619965                  0.699283                  0.645648                         0.658642            0.242297            0.312536            0.360243            0.149674               0.248901                     0.091204                 0.614444                     0.420031

## 3.2 Refresh Doctrine Answer
- Is controlled refresh justified? `False`
- Is bounded refresh preferable to naive periodic recompute? `True`

## 3.3 Step 8 Viability Answer
- Is Step 8 genuinely additive downstream? `True`
- Under what contexts is it most useful? `local balance structures with later meaningful interaction, reclaim, or nearby structural-event overlap.`
- Is it strong enough to remain a first-class source family? `True`

## 3.4 Recommended Next-Path Classification
- Primary next path: `Path C — ranking rebase`

## Downstream Usefulness Summary
- fraction_later_meaningfully_interacted: 0.6701570680628273
- fraction_later_nontrivial_breach_reclaim: 0.4607329842931937
- fraction_later_useful_event_overlap: 0.34554973821989526
- fraction_presence_improved_interpretation: 0.7539267015706806
- mean_confluence_with_other_families: 0.7958115183246073
- redundancy_assessment: genuinely_additive

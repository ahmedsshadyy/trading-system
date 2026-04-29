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
box_too_narrow_for_visible_structure    158
box_not_visually_real                    19
edge_partially_matches_chart              6
edge_matches_chart_well                   6
edge_misses_chart_reality                 3
box_too_wide_for_visible_structure        2

## Active-State Truth Audit
failure_classification
active_box_remained_truthful    104
confirm_too_late                 44
confirm_too_early                25
frozen_box_became_stale          11
lineage_fragmentation            10

## Frozen vs Refresh Comparison
                            chart_faithfulness_score  source_coherence_score  visual_plausibility_score
doctrine                                                                                               
bounded_continuity_refresh                  0.338219                0.765313                   0.632755
fixed_horizon_recompute                     0.441902                0.598794                   0.623735
frozen                                      0.200733                1.000000                   0.658499

## Coverage-Regime Comparison
       rung_id  confirmed_count  active_rows  median_confirm_latency  durable_plausible_count  fragile_plausible_count  weak_false_positive_count  strong_false_positive_count  plausible_total  false_positive_total  strong_false_positive_share  durable_plausible_share strength_alignment_status viability_alignment_status      chart_review_status  valid_contract_rung
step8e_a/mid_a              417         2150                     2.0                       34                      129                        187                          116              163                   303                     0.382838                 0.081535                        ok                         ok                too_noisy                False
step8e_a/mid_b              381         1891                     2.0                       29                      119                        172                          102              148                   274                     0.372263                 0.076115                        ok                         ok                too_noisy                False
step8e_a/mid_c              198         1029                     3.0                       20                       69                         67                           45               89                   112                     0.401786                 0.101010                        ok                         ok too_sparse_or_misaligned                False
step8e_a/mid_d              177          876                     3.0                       18                       60                         61                           39               78                   100                     0.390000                 0.101695                        ok                         ok too_sparse_or_misaligned                False
step8e_a/mid_e              212         1087                     3.0                       20                       76                         70                           47               96                   117                     0.401709                 0.094340                        ok                         ok too_sparse_or_misaligned                False
step8e_a/mid_f              198         1029                     3.0                       20                       69                         67                           45               89                   112                     0.401786                 0.101010                        ok                         ok too_sparse_or_misaligned                False
step8e_b/mid_a              373         2078                     2.0                       34                      130                        153                           96              164                   249                     0.385542                 0.091153                       bad                         ok                too_noisy                False
step8e_b/mid_b              327         1788                     2.0                       29                      118                        133                           77              147                   210                     0.366667                 0.088685                       bad                         ok                too_noisy                False
step8e_b/mid_c              175          992                     3.0                       21                       71                         50                           33               92                    83                     0.397590                 0.120000                        ok                         ok too_sparse_or_misaligned                False
step8e_b/mid_d              147          805                     3.0                       19                       58                         44                           24               77                    68                     0.352941                 0.129252                        ok                         ok too_sparse_or_misaligned                False
step8e_b/mid_e              188         1046                     3.0                       21                       75                         52                           36               96                    88                     0.409091                 0.111702                        ok                         ok too_sparse_or_misaligned                False
step8e_b/mid_f              175          992                     3.0                       21                       71                         50                           33               92                    83                     0.397590                 0.120000                        ok                         ok too_sparse_or_misaligned                False

## Ranking Disagreement
                 ranking_metric  top_n  durable_plausible  fragile_but_plausible  weak_false_positive  strong_false_positive  mean_monitor_worthiness  mean_plausibility  mean_micro_box_risk
                strength_legacy     20                  4                      6                    3                      2                 0.353125           0.481134             0.382500
range_strength_viability_legacy     20                  3                      7                    2                      1                 0.422917           0.508232             0.376667
                       strength     20                  4                      6                    1                      2                 0.356250           0.493065             0.332500
  range_strength_monitorability     20                  4                      5                    2                      2                 0.343750           0.481774             0.350833
        range_strength_semantic     20                  5                      5                    1                      2                 0.375000           0.499602             0.335000
             strength_repair_v1     20                 18                      0                    0                      0                 0.798958           0.755861             0.239167
             strength_repair_v2     20                 18                      0                    0                      0                 0.814583           0.758778             0.252500
             strength_repair_v3     20                 18                      0                    0                      0                 0.814583           0.758778             0.252500
             strength_repair_v4     20                 17                      0                    0                      0                 0.808333           0.757193             0.244167
          rb_plausibility_score     20                 18                      0                    0                      0                 0.814583           0.758778             0.252500
    rb_monitor_worthiness_score     20                 16                      1                    0                      0                 0.825000           0.751048             0.270833
    rb_boundary_relevance_score     20                 10                      8                    0                      0                 0.613542           0.675283             0.242500

## Family Comparison
                    family  rows  strength  strength_legacy  range_strength_structure  range_strength_monitorability  range_strength_semantic  range_strength_formation  range_strength_viability  range_strength_viability_legacy  strength_repair_v1  strength_repair_v2  strength_repair_v3  strength_repair_v4  rb_plausibility_score  rb_monitor_worthiness_score  rb_micro_box_risk_score  rb_boundary_relevance_score
 short_lived_high_strength    10  0.703563         0.631123                  0.752645                       0.725036                 0.677672                  0.752645                  0.702917                         0.731284            0.267169            0.345152            0.390044            0.172432               0.280362                     0.087500                 0.608333                     0.512726
long_lived_medium_strength    10  0.664239         0.574552                  0.711497                       0.686711                 0.637801                  0.711497                  0.663932                         0.680976            0.583835            0.841043            0.658660            0.528775               0.645109                     0.666667                 0.305000                     0.560637
         durable_plausible    21  0.672435         0.581234                  0.719799                       0.693087                 0.647510                  0.719799                  0.671387                         0.678628            0.679584            0.948412            0.727271            0.644144               0.750716                     0.794643                 0.243651                     0.646909
     fragile_but_plausible    75  0.653276         0.562029                  0.700038                       0.675706                 0.626954                  0.700038                  0.654498                         0.671146            0.494991            0.659732            0.587206            0.414307               0.535318                     0.436389                 0.306000                     0.556782
       weak_false_positive    20  0.661914         0.574911                  0.717761                       0.685187                 0.633429                  0.717761                  0.658706                         0.667959            0.297356            0.380013            0.416308            0.206469               0.321228                     0.173958                 0.570000                     0.485591
     strong_false_positive    36  0.644721         0.559384                  0.695672                       0.667180                 0.617702                  0.695672                  0.648025                         0.672986            0.232128            0.301711            0.349461            0.140413               0.239690                     0.067708                 0.628704                     0.424419

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
- fraction_later_meaningfully_interacted: 0.711340206185567
- fraction_later_nontrivial_breach_reclaim: 0.5154639175257731
- fraction_later_useful_event_overlap: 0.27835051546391754
- fraction_presence_improved_interpretation: 0.7628865979381443
- mean_confluence_with_other_families: 0.8144329896907216
- redundancy_assessment: genuinely_additive

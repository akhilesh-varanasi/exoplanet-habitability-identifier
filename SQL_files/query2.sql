.mode csv
.output none_habitable_only_ml_ready.csv

SELECT * FROM EPA_PLANETARY_SYSTEMS where hb_flag = 0;

.output stdout
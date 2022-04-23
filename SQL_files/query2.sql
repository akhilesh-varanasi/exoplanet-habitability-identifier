.output combine.csv

SELECT *
FROM PHL_OPTIMISTIC_INDEX AS P LEFT OUTER JOIN EPA_PLANETARY_SYSTEMS AS E
ON P.name = E.pl_name;

.output stdout
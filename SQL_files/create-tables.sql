PRAGMA foreign_keys=ON;

CREATE TABLE PHL_OPTIMISTIC_INDEX (
    Name VARCHAR(256) PRIMARY KEY,
    Type VARCHAR(256),
    Mass float,
    Radius float,
    Flux float,
    Tsurf float,
    Period float,
    Distance float,
    ESI float
);

CREATE TABLE EPA_PLANETARY_SYSTEMS (
    pl_name VARCHAR(256) PRIMARY KEY,
    hostname VARCHAR(256),
    sy_snum int,
    sy_pnum int,
    pl_orbper float,
    pl_orbsmax float,
    pl_rade float,
    pl_bmasse float,
    pl_orbeccen float,
    pl_insol float,
    pl_eqt float,
    st_teff float,
    st_rad float,
    st_mass float,
    st_met float,
    st_logg float,
    sy_gaiamag float,
    hb_flag int
);
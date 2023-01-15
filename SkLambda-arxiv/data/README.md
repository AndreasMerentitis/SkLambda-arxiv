# arxiv_collector

## Introduction

The goal is to systematically download meta-data (abstracts, etc) from the Arxiv (www.arxiv.org) using the OAI API (https://arxiv.org/help/bulk_data). The use of arxiv.org is subject to their data access policy (please read https://arxiv.org/help/robots). The repository is forked from the one developed by Wing-Kit Lee.

## Example

We need specify the start and end dates for abstract and the category (default is phyiscs:astro-ph). For example, the command

```
python data/collect.py 2015astroph.h5 -start 2015-01-01 -end 2015-12-31
python data/collect.py -c stat 2015ml.h5 -start 2015-01-01 -end 2015-12-31
```

downloads all Astro-ph abstracts appears in 2015 and save it in Pandas' HDF5 format. Note that the date is the latest modification date, which may be different from the original date of submission.

Downloading abstracts for a single year takes around a bit more than 20 minutes.

## About
* Author: Wing-Kit Lee (wingkitlee0@outlook.com)
* Extensions: Andreas Merentitis (andreas.merentitis@gmail.com)


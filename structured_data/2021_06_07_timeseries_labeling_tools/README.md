# Time series labeling tools
## Goal
This document tries to give an overview of tools that can be used for data labeling, specifically for time series.
This document was written in June 2021.

Things that were looked at are:

- anomaly annotation
- window annotation
- multivariate versus univariate
- ease of use
- cloud versus on premise

## TL;DR
Use Label Studio.
Grafana can be interesting for *live* labeling, but for all other cases Label Studio is the tool to pick.

## Interesting Tools

### Label Studio

| Variable      | Value                    |
|---------------|--------------------------|
| Tech          | mainly python            |
| Maintained    | ✔                        |
| Multivariate  | many plotted in parallel |
| GCP connector | Yes (not tested yet) |

#### Tool
In general this looks very mature:
- login page with registration, setup of projects
- setting up tasks (e.g. stretches of time series that need to be labelled)
- multivariate versus univariate. Multivariate plots must be setup during data import.
- labels can be tagged with sub tags, and with a quality rating.
- seems to have a nice community (albeit unsure how big the time series use case is)

#### Setup
There are different ways to install Label Studio.
It can be easily installed using pip, using docker,... (https://labelstud.io/guide/install.html)

When installing it in the cloud on Google Compute Engine, be aware of scopes to get it working:

> Do take care of scopes! Those are a legacy layer on top of IAM. You need to give your CE instance the following settings:
> - Cloud API access scopes
> - Allow full access to all Cloud APIs

By default an SQLite is used in the backend. 
If this would be too limiting, a PostgreSQL database can be used as well (https://labelstud.io/guide/storedata.html). 

In Label Studio users and projects can be created. 
Within these projects different *label tasks* can be created. 
Label tasks are time series segments that need to be labeled.
Once the operator is happy with the labeling, it can be submitted.

#### Import
To import data, a data template template needs to be made. 
This template is basically the configuration of how the data looks like (which columns are important), but also how the data can be viewed during labeling.
This allows for a very flexible import system.
More info on these data templates: [https://labelstud.io/templates/time_series.html](https://labelstud.io/templates/time_series.html)

The data needs to be formatted in wide format, which for certain cases might lead to sparse datasets.
On top of that, Label Studio seems to set missing values (~ sparse data) to 0, complicating importing data from a single source, for data that was not taken at the same time.
To avoid this, multiple different files can be used.

The tool can use different data sources. For an overview check: [https://labelstud.io/guide/storedata.html](https://labelstud.io/guide/storedata.html).
As an example, Label Studio can directly connect to a GCP (storage bucket): [https://labelstud.io/guide/storage.html#Google-Cloud-Storage](https://labelstud.io/guide/storage.html#Google-Cloud-Storage).
Unfortunately this did not seem to work outside of the VPC. This is probably due to a rights issue.

Data can be imported using the GUI or using an API.
Files up to 200Mb are supposed to work, although it started failing above 50Mb using the GUI.
Files that are bigger suffer from a *validation error*.

Label Studio has an API that can be used to pump data in.  ([https://labelstud.io/guide/tasks.html#Import-data-using-the-API](https://labelstud.io/guide/tasks.html#Import-data-using-the-API))

Some other tips:
- Data should be sorted on time
- Missing data: data will be set to 0 (flat line).
- Corrupt data (e.g. strings): time series just stop at the first time point it encountered a string. So even if the data afterwards is okay. There is simply no data to been seen (empty time series plot).

So be a bit careful. A small anomaly might destroy the rest of the time series. Also the sparse data part is something to keep an eye on.

#### Export
Data can be exported in different ways, JSON making most sense. There is also an API, which allows to automate the export process + parsing + dumping towards e.g. GCP.

[https://labelstud.io/templates/time_series.html#Output-format-example](https://labelstud.io/templates/time_series.html#Output-format-example)

#### Playground
Online sandbox to play around.
[https://labelstud.io/playground/](https://labelstud.io/playground/)

#### Conclusion
In general the whole interface, with users and tasks make it really interesting to work with and to organize it when having multiple users.
The tool also has a some flexibility regarding backend data storage as well as for importing data from different sources.

[https://labelstud.io/](https://labelstud.io/)

### Grafana

| Variable      | Value                    |
|---------------|--------------------------|
| Tech          | -                        |
| Maintained    | ✔                        |
| Multivariate  | many plotted in parallel |
| GCP connector | bigquery plugin          |

> The analytics platform for all your metrics

#### Tool
Grafana is a very nice and flexible tool to build dashboard of all kinds of time series data.

But it can also be used for annotation (https://grafana.com/docs/grafana/latest/dashboards/annotations/). 
Using `CTRL + click`, a simple annotation can be added, and tagged.
There is no pre filled list of tags from which an operator can pick. 
Retyping tags might (will) lead to 'typo-diversity', hampering the applicability of the tool.

Regarding this last remark, a possible interesting thread:
[https://github.com/grafana/grafana/issues/24674](https://github.com/grafana/grafana/issues/24674)

Albeit the annotation ability for Grafana is rather limited, it might be an interesting option for *live* labeling, e.g. in a factory, something happened, and the operator can annotate this directly as it happens.

#### Setup
For more information: 
[https://grafana.com/docs/grafana/latest/installation/](https://grafana.com/docs/grafana/latest/installation/)

#### Import
Grafana can use many different data sources (officially: [https://grafana.com/docs/grafana/latest/datasources/](https://grafana.com/docs/grafana/latest/datasources/) and many more using plugins: [https://grafana.com/grafana/plugins/?type=datasource](https://grafana.com/grafana/plugins/?type=datasource))).
BigQuery is supported using one of the plugins.

As Grafana is build as a dashboard tool, it is very flexible in making very different plots, combining different parameters.

#### Export
This is one of the less attractive points of Grafana for labeling.
The annotation data is stored in a Grafana database, which can be queried using the API: [https://grafana.com/docs/grafana/latest/http_api/annotations/](https://grafana.com/docs/grafana/latest/http_api/annotations/).

On the other hand annotations can be imported from various sources, so Grafana can be interesting to visualize annotation.

There is plugin that could store the data in an influxdb. However question is how mature the plugin is ([https://grafana.com/grafana/plugins/novalabs-annotations-panel/](https://grafana.com/grafana/plugins/novalabs-annotations-panel/)).

#### Conclusion
Grafana is an excellent tool to visualize time series. 
Annotation is also part of Grafana.
However, when having to label many/long time series the tool is probably limited.
Also, there seems to be a lot of freedom in the tags that are being used. 
This might bring more data garbage into the system.

Interesting tool, but for labeling only to be used for specific use cases.

## So-so tools or rising stars?

### Universal Data Tool

| Variable      | Value                    |
|---------------|--------------------------|
| Tech          | Javascript               |
| Maintained    | ✔                        |
| Multivariate  | many plotted in parallel |
| GCP connector | -                        |

#### Tool
Universal Data tool is a labeling tool developed for pictures and videos labeling.
Time series have recently been added to the Universal Data Tool, and feels at the time of writing less mature.

#### Data
Time series are not (yet) in the documentation of this tool.
So the tool might be to immature as of the time of writing.

#### Import and Export
The tool allows to import data from urls, desktop, AWS, google drive, JSON/CSV and COCO.
So no GCP connector (yet).

#### Conclusion
The tool was built with images and videos in mind. Time series have only recently been added, and the tool feels a bit immature in that perspective.

* [https://github.com/UniversalDataTool/universal-data-tool](https://github.com/UniversalDataTool/universal-data-tool)
* [https://universaldatatool.com/](https://universaldatatool.com/)
* [https://docs.universaldatatool.com/](https://docs.universaldatatool.com/)

### Trainset

| Variable      | Value                      |
| ------------- | -------------------------- |
| Tech          | nodejs                     |
| Maintained    | ✔                         |
| Multivariate  | plot 2 time series simultaneously, annotate 1 |
| GCP connector | No                         |

Trainset is a very lightweight time series annotation tool.

Official page: [https://trainset.geocene.com/](https://trainset.geocene.com/)
Github repo: [https://github.com/geocene/trainset](https://github.com/geocene/trainset) 

#### Tool
[https://github.com/Geocene/trainset](https://github.com/Geocene/trainset) show the install instructions.

On the official page, you can quickly give it a spin with some sample data.

In the tool, two different time series (a reference series and a labeling series).
Browsing through the time series is easy and intuitive.
However, only a single label can be annotated to a certain point (window).

#### Data
Data import and export can be done using csv's.

#### Conclusion
This tools is very lightweight. 
It serves up as simple page that in which data can be uploaded and annotated.
This is contrast with e.g. Label Studio in which a complete users backend is implemented.

## No Go tools

### WDK

| Variable      | Value        |
| ------------- | ------------ |
| Tech          | Matlab 2019a |
| Maintained    | ?            |
| Multivariate  |              |
| GCP connector |              |

WDK is a Matlab/Node-RED toolkit developed to work with data coming from wearables.

The toolkit comprises indeed of a nice looking time series annotation tool.

However the biggest problem is the need for proprietary software being Matlab 2019a.

[https://github.com/avenix/WDK](https://github.com/avenix/WDK)

### Curve

| Variable      | Value   |
| ------------- | ------- |
| Tech          | python2 |
| Maintained    | No      |
| Multivariate  |         |
| GCP connector |         |

- [https://github.com/baidu/Curve](https://github.com/baidu/Curve)

### TagAnomaly

| Variable      | Value     |
| ------------- | --------- |
| Tech          | R (shiny) |
| Maintained    | ?         |
| Multivariate  |           |
| GCP connector | No        |

> Anomaly detection labeling tool, specifically for multiple time series (one time series per category).

> ￼ Note: This tool was built as a part of a customer engagement, and is not maintained on a regular basis. 

Feels bloated, not so easy to use.

- [https://github.com/Microsoft/TagAnomaly](https://github.com/Microsoft/TagAnomaly)


### CrowdCurio

| Variable      | Value     |
| ------------- | --------- |
| Tech          | Javascript |
| Maintained    | ?         |
| Multivariate  | ?          |
| GCP connector | No        |

Not sure if still alive...

- [https://github.com/CrowdCurio/time-series-annotator](https://github.com/CrowdCurio/time-series-annotator)


## Other tools, possibly interesting

### hastic
Hastic ([https://hastic.io/](https://hastic.io/) is tool that that allows for easy pattern and anomaly detection.
There are two parts: hastic server and a hastic grafana app.

So in a sense, this is also a labeling tool, albeit an automated one.

As it plugs in into Grafana a lot of different data data sources can be used, e.g.

The tool was not tested, so unsure how good the pattern recognition is. 

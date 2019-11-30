#### Ingest raw yelp business attributes json file into Elastic search for Visualization and exploration purposes.

## Elasticseach

Install ElasticSearch into local machine using the 
following command

`brew tap elastic/tap`

`brew install elastic/tap/elasticsearch-full`

Once you install ElasticSearch, start the service by running this command

`elasicsearch`

Elasticseach by default runs on `localhost:9200`



## Kibana

Kibana is a visualization and query builder tool for the data in elastic search.
Install Kibana using the following commands

`brew tap elastic/tap`

`brew install elastic/tap/kibana-full`
 
Once you install ElasticSearch, start the service by running this command

`kibana`

By default kibana runs on `localhost:5601`

## Logstash

Logstash supports building pipelines and support various plugins.
In this project we are usign file input plugin and elasticsearch output
plugin to read data from `yelp business attributes json` file and write it 
into elastic search.

Configuration file need to export the data from raw json file to 
the elasticsearch is included.  

`brew tap elastic/tap`

`brew install elastic/tap/logstash-full`
 
Once you install ElasticSearch, start the pipeline by running this command

`logstash -f pipeline.conf`

`Note: path in the pipeline.conf must be provided as an absoulte path. Replace it 
with the location based on your clone path`

 
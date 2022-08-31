## To use MELT

Install the system
```
sudo mvn clean install dependency:copy-dependencies
```

Run the system
```
cd target
java -cp a-lion-1.0.jar:dependency/* org.borg.alion.EvaluationExample
```

package org.borg.alion;

import de.uni_mannheim.informatik.dws.melt.matching_base.external.docker.MatcherDockerFile;
import de.uni_mannheim.informatik.dws.melt.matching_base.external.seals.MatcherSeals;
import de.uni_mannheim.informatik.dws.melt.matching_data.Track;
import de.uni_mannheim.informatik.dws.melt.matching_data.TrackRepository;
import de.uni_mannheim.informatik.dws.melt.matching_eval.ExecutionResultSet;
import de.uni_mannheim.informatik.dws.melt.matching_eval.Executor;
import de.uni_mannheim.informatik.dws.melt.matching_eval.evaluator.EvaluatorCSV;
import de.uni_mannheim.informatik.dws.melt.matching_jena_matchers.external.matcher.SimpleStringMatcher;
import org.borg.alion.DemoPythonMatcher;
import eu.sealsproject.platform.res.domain.omt.IOntologyMatchingToolBridge;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.lang.Thread;
/**
 * Make sure docker is running.
 */
public class EvaluationExample {


    public static void main(String[] args) {

        // STEP 1: Let's initialize the matchers we want to evaluate

        // CASE 1: A matcher we can directly instantiate:
        //DemoPythonMatcher classMatcher = new DemoPythonMatcher();
	
        // CASE 2: SEALS Package
        // If you have problems with your java version, have a look at our user guide on how to manually set
        // a path to JAVA 8 for SEALS: https://dwslab.github.io/melt/matcher-packaging/seals#evaluate-and-re-use-a-seals-package-with-melt
        //File sealsFile = loadFile("simpleSealsMatcher-1.0-seals_external.zip");
        //MatcherSeals sealsMatcher = new MatcherSeals(sealsFile);

        // CASE 3: Web Docker Package
        //File dockerFile = loadFile("a-lion-1.0-web-latest.tar.gz");
	File dockerFile = new File("a-lion-1.0-web-latest.tar.gz");
        MatcherDockerFile dockerMatcher = new MatcherDockerFile("a-lion-1.0-web", dockerFile);


        // STEP 2: Run (execute) the 3 matchers to obtain an ExecutionResultSet instance

        // Let's add all tracks we want to evaluate to a list
        List<Track> tracks = new ArrayList<>();
        tracks.add(TrackRepository.Conference.V1);
        tracks.add(TrackRepository.Anatomy.Default);
        tracks.add(TrackRepository.Biodiv.Default);


        // Let's add all matchers to a map (key: matcher name, value: matcher instance)
        Map<String, IOntologyMatchingToolBridge> matchers = new HashMap<>();
        //matchers.put("Class Matcher", classMatcher);
	//        matchers.put("SEALS Matcher", sealsMatcher);
        matchers.put("Docker Matcher", dockerMatcher);

        // Let's run the matchers on the tracks:
	for (Track track: tracks) {
	    ExecutionResultSet result = Executor.run(track, matchers);
	    // Step 3: Use your favorite evaluator to interpret the result.
	    //         The results are serialized in directory: "./results".
	    EvaluatorCSV evaluatorCSV = new EvaluatorCSV(result);
	    evaluatorCSV.writeToDirectory();
	}
	
	//        ExecutionResultSet result = Executor.run(TrackRepository.Conference.V1, matchers);

        // Step 3: Use your favorite evaluator to interpret the result.
        //         The results are serialized in directory: "./results".
        //EvaluatorCSV evaluatorCSV = new EvaluatorCSV(result);
        //evaluatorCSV.writeToDirectory();
    }


    /**
     * Helper function to load files in class path that contain spaces.
     * @param fileName Name of the file.
     * @return File in case of success, else null.
     */
    private static File loadFile(String fileName){
        try {
	    return FileUtils.toFile(EvaluationExample.class.getClassLoader().getResource(fileName).toURI().toURL());
        } catch (URISyntaxException | MalformedURLException exception){
            exception.printStackTrace();
            System.out.println("Could not load file.");
            return null;
        }
    }
}

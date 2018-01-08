package divisio.example.tensorflow.cxf;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStream;
import javax.ws.rs.core.Response;

import org.apache.cxf.helpers.IOUtils;
import org.apache.cxf.jaxrs.client.WebClient;

import org.junit.BeforeClass;
import org.junit.Test;

public class TensorflowRESTIT {
    private static String endpointUrl;

    @BeforeClass
    public static void beforeClass() {
        endpointUrl = System.getProperty("service.url");
    }

    @Test
    public void testInference() throws Exception {    	
	    	//read the CSV and test prediction with each line
		int lineCounter = 1;			
		try (final BufferedReader in = new BufferedReader(new FileReader("wine_test_predicted.csv"))) {
			String line = in.readLine();//skip header			
			while ((line = in.readLine()) != null) {
				++lineCounter;
				//poor man's CSV parsing, we have only numerical values separated by ","
				if ("".equals(line.trim())) { continue; } //skip empty lines
				final String[] tokens = line.split(",");				
				if (tokens.length != 15) {
					System.err.println("Invalid number of columns (" + tokens.length + ") in line " + 
							lineCounter + ", skipping line.");
					continue;
				}		
				final WebClient client = WebClient.create(endpointUrl + 
		        		"/tensorflow/inference"
		        		+ "?wine_type=" + tokens[1]
		        		+ "&fixed_acidity=" + tokens[2]
		        		+ "&volatile_acidity=" + tokens[3]
		        		+ "&citric_acid=" + tokens[4]
		        		+ "&residual_sugar=" + tokens[5]
		        		+ "&chlorides=" + tokens[6]
		        		+ "&free_sulfur_dioxide=" + tokens[7]
		        		+ "&total_sulfur_dioxide=" + tokens[8]
		        		+ "&density=" + tokens[9]
		        		+ "&ph=" + tokens[10]
		        		+ "&sulphates=" + tokens[11]
		        		+ "&alcohol=" + tokens[12]      		
		        	);
				//compare webserver result with 
				final Response r = client.accept("text/plain").get();
		        assertEquals(Response.Status.OK.getStatusCode(), r.getStatus());
		        final String value = IOUtils.toString((InputStream)r.getEntity());
		        assertEquals("Error for test case in line " + lineCounter, 
		        		Float.parseFloat(tokens[14]), Float.parseFloat(value), 0.0001);
			}
	    }
    }
}

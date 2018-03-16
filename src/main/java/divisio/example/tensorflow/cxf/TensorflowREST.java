package divisio.example.tensorflow.cxf;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

@Path("/tensorflow")
public class TensorflowREST {
	
	/**
	 * This instance will be set by the dependency injection defined in the beans.xml
	 */
	private SavedModelBundle bundle;
	
	public void setSavedModelBundle(final SavedModelBundle bundle) {
		this.bundle = bundle; 
	}
	
	/**
	 * wraps a single float in a tensor
	 * @param f the float to wrap
	 * @return a tensor containing the float
	 */
	private static Tensor<Float> toTensor(final float f, final Collection<Tensor<?>> tensorsToClose) {
		final Tensor<Float> t = Tensors.create(f);
		if (tensorsToClose != null) {
			tensorsToClose.add(t);
		}
		return t;
	}		
	
	private static void closeTensors(final Collection<Tensor<?>> ts) {		
		for (final Tensor<?> t : ts) {
			try {
				t.close();
			} catch (final Exception e) {
				System.err.println("Error closing Tensor.");
				e.printStackTrace();
			}
		}
		ts.clear();
	}

    @GET
    @Path("/inference")
    @Produces("text/plain")
    public String runInference(@QueryParam("wine_type") float wineType, 
    		                       @QueryParam("fixed_acidity") float fixedAcidity, 
    		                       @QueryParam("volatile_acidity") float volatileAcidity, 
    		                       @QueryParam("citric_acid") float citricAcid, 
    		                       @QueryParam("residual_sugar") float residualSugar, 
    		                       @QueryParam("chlorides") float chlorides, 
    		                       @QueryParam("free_sulfur_dioxide") float freeSulfurDioxide, 
    		                       @QueryParam("total_sulfur_dioxide") float totalSulfurDioxide, 
    		                       @QueryParam("density") float density, 
    		                       @QueryParam("ph") float ph, 
    		                       @QueryParam("sulphates") float sulphates, 
    		                       @QueryParam("alcohol") float alcohol 
    		                       ) 
    {
    		final List<Tensor<?>> tensorsToClose = new ArrayList<Tensor<?>>(20);
    		
    		try {
			//run a session just like in python
	    		final List<Tensor<?>> results = bundle.session().runner()
			.feed("wine_type"           , toTensor(wineType, tensorsToClose))
			.feed("fixed_acidity"       , toTensor(fixedAcidity, tensorsToClose))
			.feed("volatile_acidity"    , toTensor(volatileAcidity, tensorsToClose))
			.feed("citric_acid"         , toTensor(citricAcid, tensorsToClose))
			.feed("residual_sugar"      , toTensor(residualSugar, tensorsToClose))
			.feed("chlorides"           , toTensor(chlorides, tensorsToClose))
			.feed("free_sulfur_dioxide" , toTensor(freeSulfurDioxide, tensorsToClose))
			.feed("total_sulfur_dioxide", toTensor(totalSulfurDioxide, tensorsToClose))
			.feed("density"             , toTensor(density, tensorsToClose))
			.feed("ph"                  , toTensor(ph, tensorsToClose))
			.feed("sulphates"           , toTensor(sulphates, tensorsToClose))
			.feed("alcohol"             , toTensor(alcohol, tensorsToClose))
			//use the saved model CLI shipping with tensorflow to determine the name
			//of the result node
			.fetch("dnn/head/logits:0")
			.run();
	    		tensorsToClose.addAll(results);
			float[][] resultValues = (float[][]) results.get(0).copyTo(new float[1][1]);
			return Float.toString(resultValues[0][0]);
    		} finally {
    			closeTensors(tensorsToClose);
    		}        
    }
}


package divisio.example.tensorflow.cxf;
import java.nio.FloatBuffer;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;

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
	 * Helper function to wrap a single float in a tensor
	 */
	private static Tensor<Float> toTensor(final float f) {
		return Tensor.create(new long[] {1}, 
				FloatBuffer.wrap(new float[] {f}));
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
		//run a session just like in python
    		final Tensor<?> result = bundle.session().runner()
		.feed("wine_type"           , toTensor(wineType))
		.feed("fixed_acidity"       , toTensor(fixedAcidity))
		.feed("volatile_acidity"    , toTensor(volatileAcidity))
		.feed("citric_acid"         , toTensor(citricAcid))
		.feed("residual_sugar"      , toTensor(residualSugar))
		.feed("chlorides"           , toTensor(chlorides))
		.feed("free_sulfur_dioxide" , toTensor(freeSulfurDioxide))
		.feed("total_sulfur_dioxide", toTensor(totalSulfurDioxide))
		.feed("density"             , toTensor(density))
		.feed("ph"                  , toTensor(ph))
		.feed("sulphates"           , toTensor(sulphates))
		.feed("alcohol"             , toTensor(alcohol))
		//use the saved model CLI shipping with tensorflow to determine the name
		//of the result node
		.fetch("dnn/head/logits:0")
		.run()
		.get(0);
		
		float[][] resultValues = (float[][]) result.copyTo(new float[1][1]);		
        return Float.toString(resultValues[0][0]);
    }
}


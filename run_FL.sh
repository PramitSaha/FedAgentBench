if [ -f "AgentPipelines/run.py" ]; then
    echo "Running AgentPipelines/run.py..."
    
    python3 AgentPipelines/run_FL_derma.py --human_requirements "I want to train binary skin cancer detection model on dermatology images." 
    # python3 AgentPipelines/run_FL_hist.py --human_requirements "I want to train binary breast cancer detection model (i.e., benign and malignant) on histopathology images."
    # python3 AgentPipelines/run_FL_xray.py --human_requirements "I want to train pneumonia detection model on Chest XRay images."
    # python3 AgentPipelines/run_FL_ultra.py --human_requirements "I want to train breast cancer classification model on breast ultrasound images."
    # python3 AgentPipelines/run_FL_fundus.py --human_requirements "I want to train diabetic retinopathy grading model on fundus images."
    # python3 AgentPipelines/run_FL_MRI.py --human_requirements "I want to train brain tumor classification model on brain MRI images."

else
    echo "Error: AgentPipelines/run.py not found!"
    exit 1
fi
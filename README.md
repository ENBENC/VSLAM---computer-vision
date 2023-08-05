# TEK5030_VSLAM
<h2>Folders</h2>
<list>
    <li>data: The videos and images used to get the results in this paper.</li>
    <li>demo: The videos and images used in the demo.</li>
    <li>pretrained_model: Patch-NetVLAD pre-trained model.</li>
    <li>imgs: Output images from the run</li>
</list>
<h2>Files:</h2>
<list>
    <li>common_lab_utils.py: Lab code relevant for a vslam-system.</li>
    <li>database.py: Code for recognition database creation.</li>
    <li>demo.py: Run demonstrations of the project.</li>
    <li>local_matcher.py: Patch-NetVLAD, calculate patch center.</li>
    <li>patch_matcher.py: Patch-NetVLAD spatialApproximator-matcher.</li>
    <li>patchnetvlad.py: Patch-NetVLAD model.</li>
    <li>real_sense_stereo_camera.py, stereo_calibration.py: Code for stereo camera (from lab) </li>
    <li>vslam.py: Original development document. demo.py is the cleaned-up version used for the presentation.</li>
</list>

<h2>Notes on how to run the code:</h2>
<list>
    <li>Pretrained-model: wget -O pittsburgh_WPCA512.pth.tar https://cloudstor.aarnet.edu.au/plus/s/WKl45MoboSyB4SH/download</li>
    <li>The required libraries (and possibly many more we don't use anymore) can be found and installed with requirements.txt. </li>
    <li>Run demo 1 with "python demo.py". Demo 1 takes a photo from the Intel Realsense stereo camera and looks for similar places in the predefined database by pressing "s". Change database video on line 161. </li>
    <li>Run demo 2 with "python demo.py 2". Demo 2 compares a predefined image with a predefined video loaded into the database. Change database video on line 161 and compared image on line 168.</li>
    <li>In case of errors, note that demo2 has not been tested after kitti was removed from stereo_calibration.py (because we do not have the stereo cameras any more).</li>
</list>
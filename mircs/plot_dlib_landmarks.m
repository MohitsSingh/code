function plot_dlib_landmarks(landmarks,markerSpec)
dlib_landmark_split;
if nargin < 2
    markerSpec = {'g-','LineWidth',1};
end
plotPolygons_open(landmarks(dlib.eyebrow_left,:), markerSpec{:});
plotPolygons_open(landmarks(dlib.eyebrow_right,:), markerSpec{:});
plotPolygons(landmarks(dlib.eye_left,:), markerSpec{:});
plotPolygons(landmarks(dlib.eye_right,:), markerSpec{:});
plotPolygons_open(landmarks(dlib.face_bottom,:), markerSpec{:});
plotPolygons(landmarks(dlib.mouth_inner,:), markerSpec{:});
plotPolygons(landmarks(dlib.mouth_outer,:), markerSpec{:});
plotPolygons_open(landmarks(dlib.nose_bottom,:), markerSpec{:});
plotPolygons_open(landmarks(dlib.nose_bridge,:), markerSpec{:});
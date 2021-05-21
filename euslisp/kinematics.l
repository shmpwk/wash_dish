#!/usr/bin/env roseus

(defun get-jacobi-end2obj-local ()
  (setq link-list (send *pr2* :link-list (send (send *pr2* :larm :end-coords) :parent)))
  (setq joint-list (send-all link-list :joint))

  (setq jacobi
	(send *pr2* :calc-jacobian-from-link-list
	      link-list 
	      :move-target (send *pr2* :larm :end-coords)
	      :rotation-axis t
	      ))
  (format-array jacobi) ;; print
  (publish-jacobi 0)
 )

(defun make-end2obj ()
  (setq l-end-coords (send *pr2* :larm :end-coords))
  (print l-end-coords)
  (send l-end-coords :assoc *init-target-obj*)
  (print l-end-coords)
  (objects (list l-end-coords *init-target-obj*))
  )
(defun make-end2obj ()
  (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
  (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
  (setq r-end-coords (send *pr2* :rarm :end-coords))
  (print r-end-coords)
  (send r-end-coords :assoc hold-obj)
  (print r-end-coords)
  (objects (list r-end-coords hold-obj))
  (setq coords (make-coords :pos (v+ (send hold-obj :worldpos)
                                     (float-vector 0 0 150))
                            ;;:rpy #f(1.54 0.6 -1.54)))
                            :rpy #f(0 0 0)))
  (send *pr2* :rarm :inverse-kinematics coords :move-target hold-obj :rotation-axis t :translation-axis nil :debug-view nil)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  )



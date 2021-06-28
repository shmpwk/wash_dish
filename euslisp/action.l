(defun l-rub ()
  (send *ri* :start-grasp *larm* :wait nil)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
  
  (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 50 (- (/ (car *init-d-xyz*) 2.5) -50) 70))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 50 (- (/ (car *init-d-xyz*) -2.5) -50) 100))
  		     :rpy #f(-1.0 0.6 -1.54))
    :rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 100 (- (/ (car *init-d-xyz*) 2.5) -50) 50))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 100 (- (/ (car *init-d-xyz*) -2.5) -50) 100))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 150 (- (/ (car *init-d-xyz*) 2.5) -50) 50))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 150 (- (/ (car *init-d-xyz*) -2.5) -50) 100))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)

)

(defun r-rub ()
  (send *ri* :stop-grasp *larm* :wait nil)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
  
  (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
  (unless
    (send *pr2* *rarm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 50 (/ (car *init-d-xyz*) 2.5) 60))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 50 (/ (car *init-d-xyz*) -2.5) 50))
  		     :rpy #f(-1.0 0.6 -1.54))
    :rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (unless
    (send *pr2* *larm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 50 (/ (car *init-d-xyz*) 2.5) 50))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (unless
    (send *pr2* *rarm* :inverse-kinematics
  	(make-coords :pos (v+ (send hold-obj :worldpos)
  			      (float-vector 50 (/ (car *init-d-xyz*) -2.5) 60))
  		     :rpy #f(-1.0 0.6 -1.54))
  	:rotation-axis :t
      :revert-if-fail nil
      :stop 10
      ;;:thre 5
      ;;:rthre 1
      )
    (publish-error "out of reach...")
    (return-from rotate-dish nil))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
 (send *ri* :wait-interpolation)
 )

(defun ready-to-regrasp ()
    (send *ri* :start-grasp *rarm* :wait t)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (send *ri* :stop-grasp *larm* :wait nil)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (print "ready to regrasp")
  )

(defun regrasp-model-larm ()
    ;;(unless box-msg
    ;; (publish-info "target point is nil")
    ;; (return-from regrasp-larm nil))
    ;;(unless (send box-msg :boxes)
    ;; (publish-info "target point is nil")
    ;; (return-from regrasp-larm nil))

    (setq regrasp-larm nil)
    (unless regrasp-larm
      (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                      jsk_recognition_msgs::BoundingBoxArray 
                                      :timeout 10000000000))
      (if box-msg
        (print box-msg)
        (setq regrasp-larm t))
      )
    (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send *init-target-obj* :worldpos)
				      (float-vector (+ (* (/ (car *init-d-xyz*) -2.5) 0.38268343236508984) 100)  (* (/ (car *init-d-xyz*) 2.5) 0.9238795325112867) 150))
			     :rpy #f(-1.0 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send *init-target-obj* :worldpos)
				      (float-vector (+ (* (/ (car *init-d-xyz*) -2.5) 0.70710678118) 100) (* (/ (car *init-d-xyz*) 2.5) 0.70710678118) 150))
			     :rpy #f(-0.77 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from regrasp-larm nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send *init-target-obj* :worldpos)
				      (float-vector (+ 150 (/ (car *init-d-xyz*) -2.8)) 0 150))
			     :rpy #f(0.0 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from regrasp-larm nil))
    
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (print "rgrasp-larm end")
    (unix:sleep 3)
  )

(defun regrasp-larm ()
    (send *ri* :stop-grasp *larm* :wait nil)
    (send *ri* :start-grasp *rarm* :wait nil)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    ;;(unless box-msg
    ;; (publish-info "target point is nil")
    ;; (return-from regrasp-larm nil))
    ;;(unless (send box-msg :boxes)
    ;; (publish-info "target point is nil")
    ;; (return-from regrasp-larm nil))

    (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                      jsk_recognition_msgs::BoundingBoxArray 
                                      :timeout 10000000000))
    (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector (* (/ (car *d_xyz*) -2) 0.38268343236508984) (* (/ (car *d_xyz*) 2) 0.9238795325112867) 0))
			     :rpy #f(-1.0 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector (* (/ (car *d_xyz*) -2) 0.70710678118) (* (/ (car *d_xyz*) 2) 0.70710678118) 0))
			     :rpy #f(-0.77 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from regrasp-larm nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector (/ (car *d_xyz*) -2) 0 0))
			     :rpy #f(0.0 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from regrasp-larm nil))
    
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (print "rgrasp-larm end")
    (unix:sleep 3)
  )

(defun ready-to-rotate ()
    (send *ri* :start-grasp :larm)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    ;;(send *ri* :move-gripper :rarm 0.007 :wait nil)
    (send *ri* :stop-grasp :rarm)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (print "ready to rotate")
  )

(defun rotate-dish-model ()
    ;;(unless msg
    ;; (publish-info "target point is nil")
    ;; (return-from rotate-dish nil))
    ;;(unless (send msg :boxes)
    ;; (publish-info "target point is nil")
    ;; (return-from rotate-dish nil))

    (setq rotate-dish nil)
    (unless rotate-dish
      (setq msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                      jsk_recognition_msgs::BoundingBoxArray 
                                      :timeout 1000000000))
      (if msg 
        (setq rotate-dish t)))

    (print "rotate dish one-shot end")
    (setq hold-obj (bounding-box->cube (elt (send msg :boxes) 0)))

    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send *init-target-obj* :worldpos)
				      (float-vector (+  (* (/ (car *init-d-xyz*) -2.5) 0.70710678118) 100) (* (/ (car *init-d-xyz*) 2.5) 0.70710678118) 150))
			     :rpy #f(-0.77 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send *init-target-obj* :worldpos)
				      (float-vector (+ (* (/ (car *init-d-xyz*) -2.5) 0.38268343236508984) 100) (* (/ (car *init-d-xyz*) 2.5) 0.9238795325112867) 150))
			     :rpy #f(-1.0 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send *init-target-obj* :worldpos)
				      (float-vector 100 (/ (car *init-d-xyz*) 2.5) 150))
			     :rpy #f(-1.54 0.5 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (unix:sleep 3)
    (print "finish rotating")
  )

(defun rotate-dish ()
    (send *ri* :move-gripper :rarm 0.02 :wait nil)
    (send *ri* :start-grasp :larm)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    ;;(unless msg
    ;; (publish-info "target point is nil")
    ;; (return-from rotate-dish nil))
    ;;(unless (send msg :boxes)
    ;; (publish-info "target point is nil")
    ;; (return-from rotate-dish nil))

    (setq msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                      jsk_recognition_msgs::BoundingBoxArray 
                                      :timeout 1000000000))
    (print "rotate dish one-shot end")
    (setq hold-obj (bounding-box->cube (elt (send msg :boxes) 0)))
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector (* (/ (car *d_xyz*) -2) 0.70710678118) (* (/ (car *d_xyz*) 2) 0.70710678118) 0))
			     :rpy (v+ (rpy-angle (send hold-obj :worldrot)) #f(-0.77 0.6 -1.54)))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector (* (/ (car *d_xyz*) -2) 0.38268343236508984) (* (/ (car *d_xyz*) 2) 0.9238795325112867) 0))
			     :rpy #f(-1.0 0.6 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector 0 (/ (car *d_xyz*) 2) 0))
			     :rpy #f(-1.54 0.5 -1.54))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10
        ;;:thre 5
        ;;:rthre 1
        ))
      (publish-error "out of reach...")
      (return-from rotate-dish nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
    (send *ri* :wait-interpolation)
    (unix:sleep 3)
    (print "finish rotating")
  )
(defun correct-r-posture ()
  (setq correct nil)
    (unless correct
      (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
      (if box-msg
        (setq correct t)))
  (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))

  (setq r-end-coords (send *pr2* :rarm :end-coords))
  (print r-end-coords)
  (send r-end-coords :assoc hold-obj)
  (print r-end-coords)
  (setq *r-end-crd* (arrow))
  (send *r-end-crd* :newcoords (send (send *pr2* :rarm :end-coords) :copy-worldcoords))
  (setq *hold-obj* (arrow))
  (send *hold-obj* :newcoords (send hold-obj :copy-worldcoords))
  (setq coords (make-coords :pos (v+ (send hold-obj :worldpos)
                                     (float-vector 0 0 0))
                            ;;:rpy #f(1.54 0.6 -1.54)))
                            :rpy #f(-1.54 0 -3.14)))
  (setq *coords* (arrow))
  (send *coords* :newcoords (send coords :copy-worldcoords))
  (objects (list *pr2* *r-end-crd* *hold-obj* *coords*))
  (send *pr2* :rarm :inverse-kinematics coords 
        :move-target hold-obj 
        :rotation-axis :z
        :translation-axis t 
        :debug-view nil
        :revert-if-fail nil
        :stop 50)
  (setq *new-r-end-crd* (arrow))
  (send *new-r-end-crd* :newcoords (send (send *pr2* :rarm :end-coords) :copy-worldcoords))
  (objects (list *pr2* *new-r-end-crd* *r-end-crd* *hold-obj* *coords*))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (setq new nil)
  (unless new
    (setq new-box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
    (if new-box-msg
      (setq new t)))
  (setq new-hold-obj (bounding-box->cube (elt (send new-box-msg :boxes) 0)))
  (setq *new-hold-obj* (arrow))
  (send *new-hold-obj* :newcoords (send new-hold-obj :copy-worldcoords))
  (objects (list *pr2* *new-hold-obj* *hold-obj* *coords*))

)
(defun correct-l-posture ()
  (setq correct nil)
    (unless correct
      (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
      (if box-msg
        (setq correct t)))
  (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))

  (setq l-end-coords (send *pr2* :larm :end-coords))
  (print l-end-coords)
  (send l-end-coords :assoc hold-obj)
  (print l-end-coords)
  (setq *l-end-crd* (arrow))
  (send *l-end-crd* :newcoords (send (send *pr2* :larm :end-coords) :copy-worldcoords))
  (setq *hold-obj* (arrow))
  (send *hold-obj* :newcoords (send hold-obj :copy-worldcoords))
  (setq coords (make-coords :pos (v+ (send hold-obj :worldpos)
                                     (float-vector 0 0 0))
                            ;;:rpy #f(1.54 0.6 -1.54)))
                            :rpy #f(-1.54 0 -3.14)))
  (setq *coords* (arrow))
  (send *coords* :newcoords (send coords :copy-worldcoords))
  (objects (list *pr2* *l-end-crd* *hold-obj* *coords*))
  (send *pr2* :larm :inverse-kinematics coords 
        :move-target hold-obj 
        :rotation-axis :z
        :translation-axis t 
        :debug-view nil
        :revert-if-fail nil
        :stop 50)
  (setq *new-l-end-crd* (arrow))
  (send *new-l-end-crd* :newcoords (send (send *pr2* :larm :end-coords) :copy-worldcoords))
  (objects (list *pr2* *new-l-end-crd* *l-end-crd* *hold-obj* *coords*))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (setq new nil)
  (unless new
    (setq new-box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
    (if new-box-msg
      (setq new t)
      (print "aaaa")))
  (setq new-hold-obj (bounding-box->cube (elt (send new-box-msg :boxes) 0)))
  (setq *new-hold-obj* (arrow))
  (send *new-hold-obj* :newcoords (send new-hold-obj :copy-worldcoords))
  (objects (list *pr2* *new-hold-obj* *hold-obj* *coords*))
)



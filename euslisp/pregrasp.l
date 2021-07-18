(defun tf-grasp-point ()
    (let*(
         (cds (when msg (ros::tf-pose->coords (send msg :pose))))
         (base->parent
           (when msg
               (send *tfl* :wait-for-transform *base-frame-id* (send msg :header :frame_id) (ros::time 0) 0.1)
               (send *tfl* :lookup-transform
                           *base-frame-id* (send msg :header :frame_id)
                           (ros::time 0)))))
    (setq *cds* cds)
    (when (and cds base->parent)
      ;;(send *tfb* :send-transform
        (send (send base->parent :copy-worldcoords) :transform cds))
      ;;      *base-frame-id* "target_object" (send msg :header :stamp))
    )
  )

(defun grasp-target-callback (msg)
  (print "clicked")
  (setq msgarr (one-shot-subscribe "/grasp_point" geometry_msgs::PoseArray))
  (dolist (msg msgarr)
    ;;(let ((target-point (sub-grasp-point))
    (let (target-point (tf-grasp-point(msg)))
      ;;(let ((target-point (bounding-box->cube msg)))
      (unless target-point
        (ros::ros-logerr "target point is nil !!")
        (return-from grasp-target-callback nil))
      (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
      (send *ri* :wait-interpolation)
      ;; gradient from grasp point to the middle of the box
      (setq grad (v- (send (bounding-box->cube msg) :worldpos)
                              (send target-point :worldpos)))
      ;; grasping toward centripetal direction
      ;;(setq *rot* (- 1.57 (atan2  (elt grad 1) (elt grad 0))))
      (setq *rot*  (atan2  (elt grad 1) (elt grad 0)))
      ;; pregrasp pos
      (setq *pre-pos* (float-vector (elt (scale 3 grad) 0) (elt (scale 3 grad) 1) 50))
      (setq *reach-pos* (float-vector (elt (scale -1.5 grad) 0) (elt (scale -1.5 grad) 1) -80))
      ;; pre grasp rotation
      ;;(setq rdm (+ (/ pi 2) (elt (elt (rpy-angle (send *cds* :rot)) 0) 1)))
      (setq rdm (elt (elt (rpy-angle (send *cds* :rot)) 1) 1))
      (setq *grasp-rot* (float-vector *rot* rdm -1.54))
      (print "actual grasp point")
      (print *grasp-rot*)
  
      (if *grasping-object-p*
          (progn
            (publish-warn "cancel grasping...")
            (send *ri* :stop-motion)
            (send *ri* :stop-grasp *rarm* :wait t)
            (setq *grasping-object-p* nil))
        (setq *grasping-object-p* t))
      (unless
        (send *pr2* *rarm* :inverse-kinematics
              ;;(make-coords :pos target-point :worldpos)
              (make-coords :pos (v+ (send target-point :worldpos)
                                    *pre-pos*)
                                    ;;(float-vector -100 0 50))
                           ;;:rpy (float-vector 0 1.2 -1.54)) ;;success(0 1.2 -1.54)
                           :rpy *grasp-rot*)
              )
        (publish-error "out of reach...")
        (send *pregrasp-point* :newcoords
              (make-coords :pos (v+ (send target-point :worldpos)
                                      *pre-pos*)
                                      ;;(float-vector -100 0 150))
                             ;;:rpy (float-vector -1.57 1.2 -1.54)))
                             :rpy *grasp-rot*))
        (send *co-ik-target* :newcoords (send target-point :copy-worldcoords))
        (objects (list *pr2* *box* *co-ik-target* *pregrasp-point* *box-arrow*))
          (progn                                          
            (setq *grasp-state* "out of reach")      
            (publish-grasp-state)
            (publish-info "grasp failed!")          
            (ros::set-param (format nil "~/judge_grasp_success") 0)) 
        (return-from grasp-target-callback nil))
      (send *pregrasp-point* :newcoords
            (make-coords :pos (v+ (send target-point :worldpos)
                                    ;;(float-vector -100 0 150))
                                    *pre-pos*)
                           ;;:rpy (float-vector 0 1.2 -1.54)))
                           :rpy *grasp-rot*))
      (send *co-ik-target* :newcoords (send target-point :copy-worldcoords))
      (objects (list *pr2* *box* *co-ik-target* *pregrasp-point* *box-arrow*))
  )))

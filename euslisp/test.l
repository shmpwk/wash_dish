#!/usr/bin/env roseus
;; pr2-tabletop-object-grasp.l
;; Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>
;; Editor: Shumpei Wakabayashi <wakabayashi@jsk.imi.i.u-tokyo.ac.jp>

(defvar *grasp-target-bounding-box-topic* "/bounding_box_marker/selected_box")
(defvar *grasp-status-topic* "/tabletop_object_grasp_status")
(defvar *base-frame-id* "/base_footprint")
(defvar *grasp-bbox* "/segmentation_decomposer_ssd/boxes")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_rviz_plugins")

(ros::roseus "pr2_tabletop_object_grasp_node")
(ros::rate 10)

(require :pr2-interface "package://pr2eus/pr2-interface.l")
;;(require "package://wash_dish/euslisp/kinematics.l")

;; set arrow
(require "package://euslisp/jskeus/eus/models/arrow-object.l")
(setq *axis* (arrow))

(defun pr2-pregrasp-pose ()
  (send *pr2* :angle-vector #f(299.712 26.8145 7.05076 49.0149 -70.8937 92.6094 -81.3427 -6.18613 -18.0939 18.0364 -20.0459 -84.4307 -131.577 -114.592 -21.328 -3 54.75))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation))

(defun publish-status (msg-type fmt) 
  (let ((color
         (case msg-type
          ('error (instance std_msgs::ColorRGBA :init
                            :r 1.0 :g 0.0 :b 0.0 :a 1.0))
          ('warn (instance std_msgs::ColorRGBA :init
                           :r 1.0 :g 1.0 :b 0.0 :a 1.0))
          (t (instance std_msgs::ColorRGBA :init
                       :r (/ 157.0 255) :g (/ 204.0 255) :b (/ 224.0 255) :a 1.0)))))
    (unless (ros::get-topic-publisher *grasp-status-topic*) 
      (ros::advertise *grasp-status-topic* jsk_rviz_plugins::OverlayText 5)
      (unix:sleep 1))
    (ros::publish *grasp-status-topic* 
                  (instance jsk_rviz_plugins::OverlayText :init
                            :width 1920 :height 400
                            :left 0 :top 0
                            :line_width 10 :text_size 50
                            :fg_color color
                            :text fmt))))

(defun publish-info (&rest args) ;;
  (let ((s (format nil (car args) (cdr args))))
    (ros::ros-info s)
    (publish-status 'info s)))
(defun publish-warn (&rest args)
  (let ((s (format nil (car args) (cdr args))))
    (ros::ros-warn s)
    (publish-status 'warn s)))
(defun publish-error (&rest args)
  (let ((s (format nil (car args) (cdr args))))
    (ros::ros-error s)
    (publish-status 'error s)))

(defun vector3-> (applier v)
  (funcall applier
         (* (send v :x) 1000.0)
         (* (send v :y) 1000.0)
         (* (send v :z) 1000.0)))

;;バウンディングボックスの端を掴む
(defun bounding-box->cube (msg)
  (progn
    (let ((cds (ros::tf-pose->coords (send msg :pose)))
          (d (vector3-> 'list (send msg :dimensions)))
          (base->parent (send *tfl* :lookup-transform
                              *base-frame-id* (send msg :header :frame_id)
                              (ros::time 0))))
      (send *tfb* :send-transform
            (send (send base->parent :copy-worldcoords)
                  :transform cds)
            *base-frame-id* "target_object" (send msg :header :stamp))
      (print base->parent)
      (setq *d_xyz* d)
      (send (apply #'make-cube d)
            :transform
            (send (send base->parent :copy-worldcoords)
                  :transform cds)))
    )
  )
 
(defun start-collect ()
  (ros::wait-for-service "/seq_data_collection/start_request")
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq res (ros::service-call "/seq_data_collection/start_request" req))
  )

(defun end-collect ()
  (ros::wait-for-service "/seq_data_collection/end_request")
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq res (ros::service-call "/seq_data_collection/end_request" req))
  )
 
(defun regrasp-larm ()
    (send *ri* :stop-grasp *larm* :wait nil)
    (setq box-msg (one-shot-subscribe "/segmentation_decomposer/boxes" 
                                      jsk_recognition_msgs::BoundingBoxArray 
                                      :timeout 1000000))
    (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				      (float-vector (* (/ (car *d_xyz*) -2) 0.70710678118) (* (/ (car *d_xyz*) 2) 0.70710678118) 0))
			     :rpy #f(-0.77 1.0 -1.54))
		:rotation-axis :t))
      (publish-error "out of reach...")
      (return-from grasp-target-callback nil))
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (unless
	(progn
	  (send *pr2* *larm* :inverse-kinematics
		(make-coords :pos (v+ (send target-obj :worldpos)
				      (float-vector (/ (car *d_xyz*) -2) 0 0))
			     :rpy #f(0 1.0 -1.54))
		:rotation-axis :t))
      (publish-error "out of reach...")
      (return-from grasp-target-callback nil))
    
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (send *ri* :start-grasp *rarm* :wait nil)
    (send *ri* :wait-interpolation)
    (unix:sleep 3)
  )


(defun rotate-dish (msg)
    (send *ri* :move-gripper :rarm 0.02 :wait nil)
    (print "rotate dish one-shot start")
    (unless msg
     (publish-info "target point is nil")
     (return-from rotate-dish nil))
    (unless (send msg :boxes)
     (publish-info "target point is nil")
     (return-from rotate-dish nil))

    (setq *target-obj* (bounding-box->cube (elt (send msg :boxes) 0)))
    (unless *target-obj*
      (publish-info "target point is nil")
      (return-from rotate-dish nil))
    (unless *grasping-object-p*
      (setq *grasping-object-p* t))
    ;;(setq box-msg (one-shot-subscribe "/segmentation_decomposer/boxes" 
    ;;                                  jsk_recognition_msgs::BoundingBoxArray 
    ;;                                  :timeout 1000000000))
    ;;(setq box-msg (one-shot-subscribe "/segmentation_decomposer/boxes" 
    ;;                                  jsk_recognition_msgs::BoundingBoxArray 
    ;;                                  :timeout 1000000000))
    ;;(while (not *has-sub*)
      (print "rotate dish one-shot end")
      (setq hold-obj (bounding-box->cube (elt (send msg :boxes) 0)))
      (unless
	  (progn
	    (send *pr2* *larm* :inverse-kinematics
	  	(make-coords :pos (v+ (send hold-obj :worldpos)
	  			      (float-vector (* (/ (car *d_xyz*) -2) 0.70710678118) (* (/ (car *d_xyz*) 2) 0.70710678118) 0))
	  		     :rpy #f(-0.77 1.0 -1.54))
	  	:rotation-axis :t))
        (publish-error "out of reach...")
        (return-from grasp-target-callback nil))
      (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
      (send *ri* :wait-interpolation)
      (unless
	  (progn
	    (send *pr2* *larm* :inverse-kinematics
	  	(make-coords :pos (v+ (send hold-obj :worldpos)
	  			      (float-vector 0 (/ (car *d_xyz*) 2) 0))
	  		     :rpy #f(-1.54 1.0 -1.54))
	  	:rotation-axis :t))
        (publish-error "out of reach...")
        (return-from grasp-target-callback nil))
      (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
      (send *ri* :wait-interpolation)
      (send *ri* :start-grasp *rarm* :wait nil)
      (send *ri* :wait-interpolation)
      (unix:sleep 3)
  )

(defun feedback ()
  (setq box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
  (setq hold-obj (bounding-box->cube (elt (send box-msg :boxes) 0)))
  (setq tilt-vector (elt (rpy-angle (send hold-obj :worldrot)) 0))
  (unless
    (progn
	  (send *pr2* *rarm* :inverse-kinematics
		(make-coords :pos (v+ (send hold-obj :worldpos)
				       (float-vector 0 (/ (car *d_xyz*) -2) 0))
			     ;;:rpy #f(1.54 1.0 -1.54))
			     ;;:rpy (float-vector 1.54 1.0 -1.54))
                 ;;:rpy (float-vector 2.38079 0.045811 3.09867)) 
			     :rpy (v- (float-vector 1.54 1.0 -1.54) (v- (float-vector (elt tilt-vector 0) (elt tilt-vector 1) (elt tilt-vector 2)) *rpy*)))
		:rotation-axis :t
        :revert-if-fail nil
        :stop 10))
    ;;(send *pr2* *rarm* :inverse-kinematics
    ;;;;  (send hold-obj :copy-worldcoords)
    ;;;;  :rotation-axis :z 
    ;;;;  :debug-view nil 
    ;;;;  :revert-if-fail nil
    ;;;;  :stop 50))
    ;;(setq tilt-vector (elt (rpy-angle (send hold-obj :worldrot)) 0))
    ;;(print (v+ (float-vector (elt tilt-vector 0) (elt tilt-vector 1) (elt tilt-vector 2)) (float-vector 1.54 1.0 -1.54)))
    ;;(print "----------------")
  	;;(make-coords :pos (v+ (send hold-obj :worldpos)
  	;;		      (float-vector 0 (/ (car *d_xyz*) -2) 0))
    ;;             :rpy #f(1.54 1 -1.54))
	;;		     ;;:rpy (v+ (float-vector (elt tilt-vector 0) (elt tilt-vector 1) (elt tilt-vector 2)) (float-vector 1.54 1.0 -1.54)))
    ;;  :rotation-axis :t
    ;;  :revert-if-fail nil
    ;;  :stop 10
    ;;  ;;:thre 5
    ;;  ;;:rthre 1
    ;;  ))
    (publish-error "out of reach...")
    (return-from feedback nil))
  
  
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (unix:sleep 3)
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
  (setq new-box-msg (one-shot-subscribe "/segmentation_decomposer_ssd/boxes" 
                                    jsk_recognition_msgs::BoundingBoxArray 
                                    :timeout 10000000000))
  (setq new-hold-obj (bounding-box->cube (elt (send new-box-msg :boxes) 0)))
  (setq *new-hold-obj* (arrow))
  (send *new-hold-obj* :newcoords (send new-hold-obj :copy-worldcoords))
  (objects (list *pr2* *new-hold-obj* *hold-obj* *coords*))

)

(defun grasp-target-callback (msg)
  (let ((target-obj (bounding-box->cube msg)))
    ;;(print target-obj)
    (setq *init-target-obj* target-obj)
    (setq rpy-vec (elt (rpy-angle (send target-obj :worldrot)) 0))
    (setq *rpy* (float-vector (elt rpy-vec 0) (elt rpy-vec 1) (elt rpy-vec 2)))
    (print "-----------")
    (if *grasping-object-p*
        (progn
          (publish-warn "cancel grasping...")
          (send *ri* :stop-motion)
          (send *ri* :stop-grasp *rarm* :wait nil)
          (send *ri* :stop-grasp *larm* :wait t)
          (setq *grasping-object-p* nil))
      (setq *grasping-object-p* t))
    (unless
	(progn
	  (send *pr2* *rarm* :inverse-kinematics
		(make-coords :pos (v+ (send target-obj :worldpos)
				       (float-vector -130 (/ (car *d_xyz*) -2.5) 50))
			     ;;:rpy #f(1.54 1.0 -1.54))
			     :rpy (float-vector 1.54 1.0 -1.54))
		:rotation-axis :t)
	;;  (send *pr2* *larm* :inverse-kinematics
	;;	(make-coords :pos (v+ (send target-obj :worldpos)
	;;			      (float-vector (+ -120 (/ (car *d_xyz*) -2)) 0 50))
	;;		     :rpy #f(0 1.0 -1.54))
	;;	:rotation-axis :t)
      )
      (publish-error "out of reach...")
      (return-from grasp-target-callback nil))
    (progn
	  (publish-info "openning gripper...")
	  (print (send target-obj :worldpos))
	  (print  *d_xyz*)
	  )
    (setq *grasp-state* "reaching")
    (publish-grasp-state)
    (publish-info "pre grasp pose...")
    (send *ri* :stop-grasp *rarm* :wait nil)
    ;;(send *ri* :stop-grasp *larm* :wait t)
    (send *ri* :wait-interpolation)
    ;;(send *pr2* :head :look-at-hand *rarm*)
    ;;(send *pr2* :head :look-at-hand *larm*)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)

    (setq *grasp-state* "grasp-approaching")
    (publish-grasp-state)
    (publish-info "reaching...")
    (send *pr2* *rarm* :move-end-pos #f(105 0 -70) :world)
    ;;(send *pr2* *larm* :move-end-pos #f(105 0 -40) :world)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    ;;(send *pr2* *rarm* :move-end-pos #f(0 0 -3) :world)
    ;;(send *pr2* *larm* :move-end-pos #f(0 0 -3) :world)
    ;;(send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
    ;;(send *ri* :wait-interpolation)

    (setq *grasp-state* "grasp-closing")
    (publish-grasp-state)
    (publish-info "grasping...")
    (let ((r-grasp-result (send *ri* :start-grasp *rarm* :wait nil))
          (l-grasp-result (send *ri* :start-grasp *larm* :wait t)))
      (send *ri* :wait-interpolation)
      (when (> l-grasp-result 20.0)
        (publish-error "failed to grasp larm: ~A" l-grasp-result)
        ;; (publish-error "failed to grasp rarm: ~A" r-grasp-result)
        (send *ri* :stop-grasp *rarm* :wait nil)
        (send *ri* :stop-grasp *larm* :wait t)
        (publish-info "back to pre grasp...")
        (pr2-pregrasp-pose)
        (return-from grasp-target-callback nil)))
    (setq *grasp-state* "picking")
    (publish-grasp-state)
    (publish-info "picking up...")
    (unix:sleep 3)
    (send *pr2* *rarm* :move-end-pos #f(0 0 150) :world)
    ;;(send *pr2* *larm* :move-end-pos #f(0 0 150) :world)
    ;;(send *pr2* :head :look-at-hand *rarm*)
    ;;(send *pr2* :head :look-at-hand *larm*)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (publish-info "grasp succeeded!"))
    
    ;;(rotate-dish)
    ;;(feedback)
    (make-end2obj)
    (make-end2obj)
    (send *ri* :stop-grasp *rarm* :wait nil)
    (pr2-pregrasp-pose)
    ;;(publish-info "click tabletop object!")
  )

(defun wait-for-grasp-target ()
  (ros::subscribe *grasp-target-bounding-box-topic*
                  jsk_recognition_msgs::BoundingBox #'grasp-target-callback)
  (publish-info "click tabletop object!"))

;;(defun wait-for-grasp-target ()
;;  (ros::subscribe *grasp-bbox*
;;                  jsk_recognition_msgs::BoundingBoxArray #'rotate-dish)
;;  (publish-info "click tabletop object!")
;;  (setq *moved-status* 2)
;;)
(defun publish-grasp-state ()
  (ros::publish "grasp_state" (instance std_msgs::String :init :data *grasp-state*))) ;change from "/grasp_state" to "grasp_state"

(defun test ()
  (setq target-frame "head_mount_kinect_rgb_optical_frame")
  (setq source-frame "base_footprint")
  (setq *tfl* (instance ros::transform-listener :init))
  (send *tfl* :wait-for-transform source-frame target-frame (ros::time 0) 10)
  (setq source->target (send *tfl* :lookup-transform target-frame source-frame (ros::time 0)))
  (print source->target)
  )

(defun demo ()
  (setq *grasping-object-p* nil)
  (setq *rarm* :rarm)
  (setq *larm* :larm)
  (setq *tfl* (instance ros::transform-listener :init))
  (setq *tfb* (instance ros::transform-broadcaster :init))
  (ros::advertise "grasp_state" std_msgs::String 1) ;;change from "/grasp_state" to "grasp_state"
  (setq *grasp-state* "waiting")
  (pr2-init)
  (pr2-pregrasp-pose)
  (wait-for-grasp-target)
  ;;(rotate-dish)
  ;;(test)
  (ros::spin))
(demo)

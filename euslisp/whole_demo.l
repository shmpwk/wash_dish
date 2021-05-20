#!/usr/bin/env roseus

(defvar *grasp-target-bounding-box-topic* "/bounding_box_marker/selected_box")
(defvar *grasp-status-topic* "/tabletop_object_grasp_status")
(defvar *base-frame-id* "/base_footprint")
(defvar *grasp-bbox* "/segmentation_decomposer_ssd/boxes")

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_rviz_plugins")

(ros::roseus "pr2_tabletop_object_grasp_node")
(ros::rate 10)

(require :pr2-interface "package://pr2eus/pr2-interface.l")
(load "package://wash_dish/euslisp/action.l")
(load "angle-vectors.l")
;; set arrow
(require "package://euslisp/jskeus/eus/models/arrow-object.l")
(setq *axis* (arrow))

(defun pr2-kitchen-pose ()
  (send *pr2* :angle-vector *base-pose*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation))

(defun rm-leftover ()
  ;; pre pose
  (send *pr2* :angle-vector *rm1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; tilt dish
  (send *pr2* :angle-vector *rm2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; throw away
  (send *pr2* :angle-vector *rm1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; end pose
  (send *pr2* :angle-vector *rm1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  )

(defun rinse ()
  ;; reach handle
  (send *pr2* :angle-vector *rs1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; start water
  (send *pr2* :angle-vector *rs2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; tilt dish
  (send *pr2* :angle-vector *rs3*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; stop water
  (send *pr2* :angle-vector *rs4*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; end pose
  (send *pr2* :angle-vector *rs5*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  )

(defun scrub ()
  ;;; open hand
  ;(send *pr2* :angle-vector *rm1*)
  ;(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  ;(send *ri* :wait-interpolation)
  ;;; grub a sponge
  ;(send *pr2* :angle-vector *rm1*)
  ;(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  ;(send *ri* :wait-interpolation)
  ;; start scrub pose
  (send *pr2* :angle-vector *ws1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; end scrub pose
  (send *pr2* :angle-vector *ws2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; put a sponge
  (send *pr2* :angle-vector *ws3*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *pr2* :angle-vector *ws2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *pr2* :angle-vector *ws1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *pr2* :angle-vector *ws2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; put a sponge
  (send *pr2* :angle-vector *ws3*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;;; open hand 
  ;(send *pr2* :angle-vector *rm1*)
  ;(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  ;(send *ri* :wait-interpolation)
  ;;; end pose
  ;(send *pr2* :angle-vector *rm1*)
  ;(send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  ;(send *ri* :wait-interpolation)
  )

(defun put-dish ()
  ;; reach dish drainer
  (send *pr2* :angle-vector *put1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  ;; put dish
  (send *pr2* :angle-vector *put2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  ;; release dish
  (send *pr2* :angle-vector *put3*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)

  (send *ri* :stop-grasp :larm :wait t)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  ;; end pose
  (send *pr2* :angle-vector *put2*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *pr2* :angle-vector *put1*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  )

(defun demo ()
  (pr2-init)
  (send *ri* :start-grasp :larm :wait t)
  ;;(pr2-kitchen-pose)
  ;;(rm-leftovers)
  ;;(rinse)
  ;;(scrub)
  ;;(rinse)
  (rinse)
  (scrub)
  (rinse)
  (put-dish)
  )
  

create database face_recognition;
-- use attendence_system;
use face_recognition;

SET SQL_SAFE_UPDATES = 0;


CREATE TABLE students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    image_path VARCHAR(255) NOT NULL
);

CREATE TABLE employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    image_path VARCHAR(255) NOT NULL
);


CREATE TABLE student_attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    `IN` DATETIME NOT NULL,
    `OUT` DATETIME DEFAULT NULL,
    Status VARCHAR(10) NOT NULL
);

CREATE TABLE employee_attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    `IN` DATETIME NOT NULL,
    `OUT` DATETIME DEFAULT NULL,
    Status VARCHAR(10) NOT NULL
);



DELETE FROM employees;
DELETE FROM students;

DELETE FROM employee_attendance;
DELETE FROM attendance_summary;


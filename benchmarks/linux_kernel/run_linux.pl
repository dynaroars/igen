#!/usr/bin/perl

use strict;
use warnings;
use Time::HiRes;
use Math::BigFloat;
Math::BigFloat->precision(-3);

my $SUT_NAME="linux_kernel";

my @filesArr=("0988c4c.c", "218ad12.c", "60e233a.c", "76baeeb.c", "c708c57.c", "e39363a.c", 
              "f7ab9b4.c", "0dc77b6.c", "221ac32.c", "6252547.c", "7acf6cd.c", "d530db0.c", 
              "e67bc51.c", "0f8f809.c", "30e0532.c", "63878ac.c", "8c82962.c", "d549f55.c", 
              "eb91f1d.c", "1c17e4d.c", "36855dc.c", "657e964.c", "91ea820.c", "d6c7e11.c", 
              "ee3f34e.c", "1f758a4.c", "472a474.c", "6651791.c", "ae249b5.c", "d7e9711.c", 
              "f3d83e2.c", "208d898.c", "51fd36f.c", "6e2b757.c", "bc8cec0.c", "e1fbd92.c", "f48ec1d.c");

my $SUT_DIR="/home/ugur/igen/benchmarks/linux_kernel";
my $SRC_DIR="$SUT_DIR/src";
my $TEST_DIR="$SUT_DIR/test";

my $configIn=$ARGV[0];
my $confSpaceFile="$SUT_DIR/linux_kernel.dom";
my $cc = "gcc -std=c99 -w -o temp.out";

sub prepare($){
   my $conf=shift;
   my $result="";
   my @options=split(",", $conf);
   foreach my $o (@options){
       my @pair=split(" ", $o);
       if($pair[1] eq "on"){ $result=$result." -D".$pair[0];}
   }
   return $result;
}

sub buildSUT($){
   # Configure
   my $config=shift;
   my $start = Time::HiRes::time();
   my $logFile = "$TEST_DIR/build.$start.log";
   my $errLogFile = "$TEST_DIR/build.$start.err.log";
   `echo "---Configuration:$config,Start_Time:$start\n" > $logFile 2>&1`;
   foreach my $file (@filesArr){
      if (system("cd $SRC_DIR && $cc $config $file >> $logFile 2>&1") == 0) {
         if (system("$SRC_DIR/temp.out >> $logFile 2>&1") == 0) {
	     `echo $file.succes>>$errLogFile`;
	     `rm $SRC_DIR/temp.out >> $logFile`;
         } else {
            `echo $file.runtimeFailure>>$errLogFile`;
         }
         
      } else {
         `echo $file.compiletimeFailure>>$errLogFile`;
      }
   }
   my $time = Math::BigFloat->new(Time::HiRes::time() - $start);
   `echo "\n---Test Completed, Total Time:$time---\n" >> $logFile 2>&1`;

   print($errLogFile)
}

buildSUT(prepare($configIn));

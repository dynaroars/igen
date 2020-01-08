#!/usr/bin/perl

use strict;
use warnings;
use Time::HiRes;
use HTML::TableExtract;

no warnings;
no warnings "all";
    
my $SUT_DIR="/home/ugur/ack";
my $config=$ARGV[0];
my $ARC_DIR="$SUT_DIR/archive";
my $coverageFile;
my $settings="";

sub clean(){
   #`rm -f -r $SUT_DIR/archive`;
   `rm -fr $SUT_DIR/cover_db`;
   #`mkdir $SUT_DIR/archive`;
	 `rm -f $ARC_DIR/build-*`;
	
   my @options = split(",",$config);
   foreach my $opt (@options){
   		chomp($opt);	
      my @configSettings = split(" ",$opt);
      if ( index($configSettings[1], '-NA-') == -1) {
      	$settings = $settings." ".$configSettings[1];
      }
   }
}

sub run(){
   my $start = Time::HiRes::time();
   my $buildFile="$ARC_DIR/build-$start.txt";
   $coverageFile="$ARC_DIR/coverage-$start.txt";
   `echo $settings > $buildFile`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings "[0-9]" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings "hello world" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --version >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --help >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --help-types >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --bar >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --thpppt "hello world" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --nofilter "hello world" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --create-ackrc "hello world" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --type-del xml "hello world" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --ignore-dir=/hello-world/c "hello world" hello-world/ >> $buildFile 2>&1`;
   `cd $SUT_DIR && perl -MDevel::Cover ack.pl $settings --ignore-file=/hello-world/g/go.go "hello world" hello-world/ >> $buildFile 2>&1`;  
   `cd $SUT_DIR && cover >> $buildFile 2>&1`;
}

sub harvest(){
   my $te = HTML::TableExtract->new;
   $te->parse_file("$SUT_DIR/cover_db/ack-pl.html");
   my @tables = $te->tables;
   my $counter = 0;
   for my $row ($tables[1]->rows) {
      no warnings 'uninitialized';
      if (length $row){
      	if ($tables[1]->cell($counter,1) > 0){
         		my $lineNumber=$tables[1]->cell($counter,0);
            `echo $lineNumber >> $coverageFile 2>&1`;
         }
         $counter++;
      }
   }
}

clean();
run();
harvest();
print $coverageFile;

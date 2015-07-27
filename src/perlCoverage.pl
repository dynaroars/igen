#!/usr/bin/perl

use strict;
use warnings;
use Time::HiRes;
use HTML::TableExtract;

no warnings;
no warnings "all";

my $igenIn=$ARGV[0];
my $SUT_DIR="/fs/buzz/ukoc/ppt";

#To extract program name
my($prog_name) = $igenIn =~ m/@@@(.*)\s/;

# Replacing @@@ with perl -MDevel::Cover ppt-0.14/bin/
substr($igenIn, index($igenIn, '@@@'), 3, 'perl -MDevel::Cover ppt-0.14/bin/');

my $start = Time::HiRes::time(); # just to diffirentiate files
my $covFile="$SUT_DIR/archive/cov-$start.txt";
my $logFile="$SUT_DIR/archive/test-$start.log";

#running program
`echo "Running command: $igenIn \nTime: $start" > $logFile 2>&1`;
`$igenIn >> $logFile 2>&1`;
`cover >> $logFile 2>&1`;

#html parsing
my $te = HTML::TableExtract->new;
$te->parse_file("cover_db/ppt-0-14-bin-$prog_name.html");
my @tables = $te->tables;
my $counter = 0;
for my $row ($tables[1]->rows) {
   no warnings 'uninitialized';
   if (length $row){
      if ($tables[1]->cell($counter,1) > 0){
         my $lineNumber=$tables[1]->cell($counter,0);
         `echo $lineNumber >> $covFile 2>&1`;
      }
      $counter++;
   }
}

`rm -fr cover_db/`; #cleaning

print $covFile; #printing coverage file name

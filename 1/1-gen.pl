#!/usr/bin/perl

my $n = @ARGV[0];
my $range = @ARGV[1];
$n = 32 if !$n;
$range = 100 if !$range;

open OUT, ">input.txt";
print OUT "$n\n";
my $sum = int(rand($range));
print OUT $sum;
for ($i = 1; $i < $n; $i++) {
    my $val = int(rand($range));
    $sum += $val;
    print OUT " " . $val;
}
print OUT "\n";
close OUT;

print "Sum: $sum\n";
